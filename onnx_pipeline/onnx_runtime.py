from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ONNXModelBundle:
    model_dir: Path
    encoder_path: str | None = None
    decoder_path: str | None = None
    tokenizers_dir: str | None = None
    manifest_path: str | None = None
    provider: str = "CPUExecutionProvider"

    @property
    def manifest(self) -> dict[str, Any]:
        if self.manifest_path is None:
            return {}
        try:
            return json.loads(Path(self.manifest_path).read_text(encoding="utf-8"))
        except Exception:
            return {}


class ONNXRuntimeTTS:
    def __init__(self, model_dir: str | Path, provider: str | None = None):
        self.model_dir = Path(model_dir)
        self.provider = provider or "CPUExecutionProvider"
        self.bundle = self._discover_bundle()

    def _discover_bundle(self) -> ONNXModelBundle:
        model_dir = self.model_dir
        manifest_path = model_dir / "export_manifest.json"
        bundle = ONNXModelBundle(model_dir=model_dir, manifest_path=str(manifest_path), provider=self.provider)

        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            bundle.encoder_path = manifest.get("encoder_model")
            bundle.decoder_path = manifest.get("decoder_model")
            bundle.tokenizers_dir = manifest.get("tokenizers_dir")

        if bundle.encoder_path is None:
            for pattern in ("*encoder*.onnx", "*encoder*.ort", "*.onnx"):
                matches = sorted(model_dir.glob(pattern))
                if matches:
                    bundle.encoder_path = str(matches[0])
                    break
        if bundle.decoder_path is None:
            for pattern in ("*decoder*.onnx", "*decoder*.ort", "*.onnx"):
                matches = sorted(model_dir.glob(pattern))
                if matches:
                    bundle.decoder_path = str(matches[0])
                    break
        return bundle

    def is_ready(self) -> bool:
        if self.bundle.encoder_path is None and self.bundle.decoder_path is None:
            return False
        return True

    def generate_voice_clone(self, *, text: str, ref_audio: str, ref_text: str, language: str = "English"):
        raise NotImplementedError(
            "The ONNX export path is not yet available for this model package. "
            "You need a model implementation that exposes ONNX graph(s), or an exporter that produces compatible ORT artifacts."
        )
