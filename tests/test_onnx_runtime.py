import logging
from pathlib import Path

import numpy as np
import pytest

from onnx_pipeline.onnx_runtime import ONNXRuntimeTTS


def test_code_predictor_sampling_excludes_special_tokens():
    runtime = ONNXRuntimeTTS.__new__(ONNXRuntimeTTS)
    runtime.model_config = {"code_predictor": {"vocab_size": 2048}}
    logits = np.zeros((1, 1, 3072), dtype=np.float32)
    logits[0, 0, 2149] = 100.0
    logits[0, 0, 137] = 1.0

    assert runtime._sample_code_predictor_token(logits) == 137


def test_talker_sampling_excludes_special_tokens_but_allows_eos():
    runtime = ONNXRuntimeTTS.__new__(ONNXRuntimeTTS)
    runtime.model_config = {"talker": {"codec_eos_token_id": 2150}}
    logits = np.zeros((1, 1, 3072), dtype=np.float32)
    logits[0, 0, 2149] = 100.0
    logits[0, 0, 137] = 1.0
    assert runtime._sample_talker_codec_token(logits, allow_eos=False) == 137

    logits[0, 0, 2150] = 101.0
    assert runtime._sample_talker_codec_token(logits, allow_eos=True) == 2150


def test_onnx_runtime_discovers_bundle_and_generates_audio():
    model_dir = Path(__file__).resolve().parents[1] / "onnx_models"

    runtime = ONNXRuntimeTTS(model_dir)
    assert runtime.is_ready() is True

    wavs, sample_rate = runtime.generate_voice_clone(
        text="hello world",
        ref_audio=str(model_dir / "embeddings" / "text_embedding.npy"),
        ref_text="hello world",
        language="English",
    )

    assert len(wavs) == 1
    assert wavs[0].size > 0
    assert sample_rate == 24000


def test_onnx_runtime_logs_loaded_models_and_successful_inference(caplog):
    model_dir = Path(__file__).resolve().parents[1] / "onnx_models"

    with caplog.at_level(logging.INFO, logger="onnx_pipeline.onnx_runtime"):
        runtime = ONNXRuntimeTTS(model_dir)
        runtime.generate_voice_clone(
            text="hello world",
            ref_audio=str(model_dir / "embeddings" / "text_embedding.npy"),
            ref_text="hello world",
            language="English",
        )

    assert "Loaded ONNX model" in caplog.text
    assert "Running vocoder model=vocoder" in caplog.text
    assert "ONNX inference failed" not in caplog.text
