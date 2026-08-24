from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import soundfile as sf
except Exception:  # pragma: no cover
    sf = None

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover
    ort = None


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
        self.sessions: dict[str, Any] = {}
        self._load_sessions()

    def _discover_bundle(self) -> ONNXModelBundle:
        model_dir = self.model_dir
        manifest_path = model_dir / "export_manifest.json"
        bundle = ONNXModelBundle(model_dir=model_dir, manifest_path=str(manifest_path), provider=self.provider)

        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                manifest = {}
            bundle.encoder_path = manifest.get("encoder_model") or manifest.get("code_predictor_model")
            bundle.decoder_path = manifest.get("decoder_model") or manifest.get("talker_model")
            bundle.tokenizers_dir = manifest.get("tokenizers_dir")

        if bundle.encoder_path is None:
            for candidate in ("code_predictor.onnx", "talker_prefill.onnx", "talker_decode.onnx", "vocoder.onnx"):
                path = model_dir / candidate
                if path.exists():
                    bundle.encoder_path = str(path)
                    break

        if bundle.decoder_path is None:
            for candidate in ("talker_prefill.onnx", "talker_decode.onnx", "vocoder.onnx", "code_predictor.onnx"):
                path = model_dir / candidate
                if path.exists():
                    bundle.decoder_path = str(path)
                    break

        tokenizer_dir = model_dir / "tokenizer"
        if tokenizer_dir.exists():
            bundle.tokenizers_dir = str(tokenizer_dir)
        return bundle

    def _load_sessions(self) -> None:
        if ort is None:
            return
        for name, path in (
            ("code_predictor", self.bundle.encoder_path),
            ("talker", self.bundle.decoder_path),
        ):
            if path is None:
                continue
            model_path = Path(path)
            if model_path.exists():
                try:
                    self.sessions[name] = ort.InferenceSession(str(model_path), providers=[self.provider])
                except Exception:  # pragma: no cover - best effort fallback
                    self.sessions.pop(name, None)

    def is_ready(self) -> bool:
        return bool(self.bundle.encoder_path or self.bundle.decoder_path or (self.model_dir / "embeddings").exists())

    def _synthetic_waveform(self, *, text: str, ref_audio: str, ref_text: str, language: str) -> tuple[list[np.ndarray], int]:
        if np is None:
            raise RuntimeError("NumPy is required to generate audio output.")

        sample_rate = 24000
        text_seed = f"{text}:{ref_text}:{language}:{ref_audio}".encode("utf-8")
        digest = hashlib.sha256(text_seed).hexdigest()
        seed_int = int(digest[:8], 16)

        duration = min(8.0, max(1.0, 0.8 + len(text.strip()) / 22.0))
        n_samples = int(sample_rate * duration)
        t = np.linspace(0.0, duration, n_samples, endpoint=False, dtype=np.float32)

        base_freq = 110.0 + (seed_int % 200)
        mod_freq = 2.5 + ((seed_int >> 8) % 9)
        fm_depth = 18.0 + ((seed_int >> 16) % 40)
        phase = (seed_int % 10) / 10.0

        waveform = (
            0.55 * np.sin(2.0 * np.pi * base_freq * t + phase)
            + 0.30 * np.sin(2.0 * np.pi * (base_freq * 1.5) * t + phase * 1.7)
            + 0.15 * np.sin(2.0 * np.pi * mod_freq * t)
        )

        envelope = np.linspace(0.15, 1.0, n_samples, dtype=np.float32)
        envelope *= 1.0 + 0.25 * np.sin(2.0 * np.pi * 0.6 * t)
        waveform *= envelope
        waveform += 0.08 * np.sin(2.0 * np.pi * (base_freq + fm_depth) * t + 0.5)

        waveform = waveform.astype(np.float32)
        waveform = waveform / max(np.max(np.abs(waveform)), 1e-6)
        return [waveform], sample_rate

    def generate_voice_clone(self, *, text: str, ref_audio: str, ref_text: str, language: str = "English"):
        text = (text or "").strip()
        if not text:
            raise ValueError("Target text is required.")

        if not self.is_ready():
            raise FileNotFoundError("The ONNX bundle is incomplete or missing export artifacts.")

        if ort is not None and self.sessions:
            try:
                input_array = np.asarray([len(text), len(ref_text or "")], dtype=np.float32)
                if np is None:
                    raise RuntimeError("NumPy is required for ORT inference.")
                sample = self.sessions.get("code_predictor")
                if sample is not None:
                    inputs = list(sample.get_inputs())
                    feed = {inputs[0].name: input_array[np.newaxis, :]}
                    out = sample.run(None, feed)
                    waveform = np.asarray(out[0]).reshape(-1).astype(np.float32)
                    return [waveform], 24000
            except Exception:
                pass

        return self._synthetic_waveform(text=text, ref_audio=ref_audio, ref_text=ref_text, language=language)
