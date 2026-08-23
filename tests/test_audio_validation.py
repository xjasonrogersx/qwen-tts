from pathlib import Path

import numpy as np
import soundfile as sf

from tts import resolve_reference_audio


def _write_audio(path: Path, *, sample_rate: int = 22050, seconds: float = 0.2) -> None:
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    audio = np.stack([
        np.sin(2 * np.pi * 220 * t),
        np.sin(2 * np.pi * 330 * t),
    ], axis=1)
    sf.write(path, audio, sample_rate)


def test_resolve_reference_audio_returns_first_valid_file(tmp_path):
    valid_1 = tmp_path / "valid_a.wav"
    valid_2 = tmp_path / "valid_b.wav"
    _write_audio(valid_1)
    _write_audio(valid_2)

    resolved = resolve_reference_audio([valid_1, valid_2])

    assert resolved == str(valid_1)


def test_resolve_reference_audio_skips_missing_files(tmp_path):
    missing = tmp_path / "missing.wav"
    valid = tmp_path / "good.wav"
    _write_audio(valid)

    resolved = resolve_reference_audio([missing, valid])

    assert resolved == str(valid)
