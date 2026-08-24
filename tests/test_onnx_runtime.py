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
