# Qwen TTS


```
docker run -it --name tts -p 8000:8000 -v ubuntu:24.04 

apt install sox

pip install -r requirements.txt --break-system-packages

```

This repo supports two generation paths:

- PyTorch path via [server.py](server.py)
- ONNX path via [server_onnx.py](server_onnx.py) and [onnx_pipeline/onnx_runtime.py](onnx_pipeline/onnx_runtime.py)

## ONNX model bundle

The ONNX bundle should live under the workspace folder:

- [onnx_models](onnx_models)

A compatible bundle typically contains files such as:

- `onnx_models/code_predictor.onnx`
- `onnx_models/talker_prefill.onnx`
- `onnx_models/talker_decode.onnx`
- `onnx_models/vocoder.onnx`
- `onnx_models/embeddings/`
- `onnx_models/tokenizer/`

The downloaded model from Hugging Face is placed in the same structure and is recognized by the repo as an ONNX bundle.

## Important compatibility note

The current implementation in [onnx_pipeline/onnx_runtime.py](onnx_pipeline/onnx_runtime.py) is not a complete ONNX inference runtime yet.

It does this:

- discovers the ONNX files in the model directory
- checks whether an ONNX bundle exists
- but `generate_voice_clone()` is still a stub and raises `NotImplementedError`

So, downloading the ONNX files alone does not make the ONNX path runnable in its current form. The repo still needs a real ONNX graph execution implementation or a compatible exported model API that matches the expected `generate_voice_clone()` contract.


### 3) Start the working PyTorch server

This is the currently working path in this repo:

```bash
python3 server.py
```

Then open the served page in a browser and create a character as usual.

### 4) Start the ONNX server

If you have a complete ONNX runtime implementation for your model bundle, start:

```bash
python3 server_onnx.py
```

The ONNX path expects the model files under [onnx_models](onnx_models), but with the current code this will fail at generation time unless the inference implementation is added.

## Model source

https://huggingface.co/elbruno/Qwen3-TTS-12Hz-1.7B-CustomVoice-ONNX

## Troubleshooting

- If the model folder is missing: create [onnx_models](onnx_models) and copy the downloaded ONNX files into it.
- If generation fails with `NotImplementedError`: this is the expected state for the current ONNX runtime stub.
- If the PyTorch version is preferred: use [server.py](server.py) instead of the ONNX server.

