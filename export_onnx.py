#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a Qwen TTS model to ONNX if the installed library supports it.")
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base", help="Model ID or local path to export.")
    parser.add_argument("--output-dir", default="./onnx_models", help="Directory to store exported ONNX artifacts.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Target device for export.")
    parser.add_argument("--optimize", action="store_true", help="Enable ONNX optimizations when supported.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from qwen_tts import Qwen3TTSModel
    except Exception as exc:  # pragma: no cover
        print(f"qwen_tts is not installed or cannot be imported: {exc}", file=sys.stderr)
        return 1

    try:
        model = Qwen3TTSModel.from_pretrained(args.model, device_map=args.device)
    except Exception as exc:  # pragma: no cover
        print(f"Unable to load model {args.model!r}: {exc}", file=sys.stderr)
        return 1

    export_candidates = [
        "to_onnx",
        "export_onnx",
        "export",
        "onnx_export",
    ]
    exporter = None
    for candidate in export_candidates:
        method = getattr(model, candidate, None)
        if callable(method):
            exporter = method
            break

    if exporter is None:
        known_methods = [name for name in dir(model) if "onnx" in name.lower() or "export" in name.lower()]
        print(
            "ONNX export is not available with the installed Qwen TTS package. "
            "This model version only exposes the PyTorch generation path.",
            file=sys.stderr,
        )
        print(f"Detected export-related methods: {known_methods or 'none'}", file=sys.stderr)
        print(
            "Install a Qwen TTS build that exposes an ONNX export API, or export from a supported upstream implementation.",
            file=sys.stderr,
        )
        return 1

    export_manifest = {
        "model": args.model,
        "output_dir": str(output_dir),
        "device": args.device,
        "optimized": bool(args.optimize),
    }
    (output_dir / "export_manifest.json").write_text(json.dumps(export_manifest, indent=2), encoding="utf-8")

    print(f"Found ONNX export method: {exporter.__name__}")
    print(f"Exporting to {output_dir} ...")

    try:
        if args.optimize:
            result = exporter(output_dir, optimize=True)
        else:
            result = exporter(output_dir)
    except TypeError:
        result = exporter(output_dir)

    print("Export finished.")
    if isinstance(result, (str, Path)):
        print(f"Result: {result}")
    elif isinstance(result, dict):
        print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
