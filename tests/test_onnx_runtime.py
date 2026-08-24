import logging
from pathlib import Path

from onnx_pipeline.onnx_runtime import ONNXRuntimeTTS


def test_onnx_runtime_discovers_bundle_and_generates_stub_audio():
    model_dir = Path(__file__).resolve().parents[1] / "onnx_models"

    runtime = ONNXRuntimeTTS(model_dir)
    assert runtime.is_ready() is True

    wavs, sr = runtime.generate_voice_clone(
        text="hello world",
        ref_audio=str(model_dir / "embeddings" / "text_embedding.npy"),
        ref_text="hello world",
        language="English",
    )

    assert isinstance(wavs, list)
    assert len(wavs) >= 1
    assert sr > 0


def test_onnx_runtime_logs_loaded_models_and_inference_shapes(caplog):
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
    assert "Running ONNX inference" in caplog.text
    assert "input_shapes" in caplog.text
    assert "output_shapes" in caplog.text
