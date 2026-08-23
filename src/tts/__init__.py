from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def validate_audio_file(path: str | Path) -> tuple[bool, str | None]:
    """Return whether a reference clip is readable and safe for Qwen voice cloning."""
    file_path = Path(path)
    if not file_path.exists():
        return False, f"Audio file not found: {file_path}"

    try:
        info = sf.info(str(file_path))
    except Exception as exc:  # pragma: no cover - defensive against corrupted inputs
        return False, f"Could not inspect audio file: {exc}"

    if info.frames <= 0:
        return False, f"Audio file has no frames: {file_path}"
    if info.channels <= 0 or info.samplerate <= 0:
        return False, f"Audio file has invalid metadata: {file_path}"

    try:
        data, sample_rate = sf.read(str(file_path), always_2d=False)
    except ValueError as exc:
        return False, f"Audio file is unreadable or malformed: {exc}"
    except Exception as exc:  # pragma: no cover - defensive against corrupted inputs
        return False, f"Failed to read audio file: {exc}"

    if data.size == 0:
        return False, f"Audio file is empty: {file_path}"

    signal = np.asarray(data)
    if signal.ndim == 0 or signal.size == 0:
        return False, f"Audio file produced no samples: {file_path}"
    if sample_rate <= 0:
        return False, f"Audio file has invalid sample rate: {sample_rate}"

    return True, None


def resolve_reference_audio(candidates: list[str | Path]) -> str:
    """Select the first readable reference clip; skip malformed or corrupt files."""
    for candidate in candidates:
        ok, reason = validate_audio_file(candidate)
        if ok:
            return str(candidate)
        print(f"Skipping invalid audio candidate {candidate!s}: {reason}")

    raise FileNotFoundError("No valid reference audio files were found. Check the referenced clips.")


def main() -> None:
    import torch
    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        device_map="cpu",
    )

    candidates = [
        "audiomass-output.wav",
        "woman_speaking.flac",
    ]
    ref_audio = resolve_reference_audio(candidates)
    ref_text = (
        "wish we didn't have all these empty stores.  thats the thing.  and when people walk into town and "
        "see empty stores, they think theres nothing much here"
    )

    target_text = (
        "It wasn't just petrol. It smelled like my grandmother's kitchen when a pan of sugar burns on the "
        "stove. You could taste it in the back of your throat for miles."
    )

    wavs, sr = model.generate_voice_clone(
        text=target_text,
        language="English",
        ref_audio=ref_audio,
        ref_text=ref_text,
    )

    sf.write("output_voice_clone.wav", wavs[0], sr)


__all__ = ["main", "resolve_reference_audio", "validate_audio_file"]
