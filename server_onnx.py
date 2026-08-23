#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from server import (
    CHARACTERS_DIR,
    HTML_PAGE,
    create_character,
    list_characters,
    load_character,
    sanitize_filename,
    slugify,
    validate_audio_file,
)

ROOT = Path(__file__).resolve().parent

try:
    from onnx_pipeline.onnx_runtime import ONNXRuntimeTTS
except Exception:  # pragma: no cover
    ONNXRuntimeTTS = None


MODEL = None


def get_onnx_model():
    global MODEL
    if MODEL is not None:
        return MODEL
    if ONNXRuntimeTTS is None:
        raise RuntimeError("onnx_pipeline is not available in this environment.")
    model_dir = ROOT / "onnx_models"
    if not model_dir.exists():
        raise FileNotFoundError(
            "No ONNX model bundle was found. Export a model with export_onnx.py before starting server_onnx.py."
        )
    MODEL = ONNXRuntimeTTS(model_dir=model_dir)
    if not MODEL.is_ready():
        raise FileNotFoundError("The ONNX bundle is incomplete or missing export artifacts.")
    return MODEL


def generate_voice(character_id: str, target_text: str, language: str = "English") -> str:
    target_dir, metadata = load_character(character_id)
    ref_audio = target_dir / metadata.get("ref_audio", "")
    if not ref_audio.exists():
        raise FileNotFoundError(f"Reference clip missing for character: {character_id}")

    ok, reason = validate_audio_file(ref_audio)
    if not ok:
        raise ValueError(f"Saved reference clip is invalid: {reason}")

    text = (target_text or "").strip()
    if not text:
        raise ValueError("Target text is required.")

    model = get_onnx_model()
    wavs, sr = model.generate_voice_clone(
        text=text,
        ref_audio=str(ref_audio),
        ref_text=metadata.get("ref_text", ""),
        language=language,
    )

    output_name = f"generated_onnx_{int(time.time() * 1000)}.wav"
    output_path = target_dir / output_name
    import soundfile as sf
    sf.write(output_path, wavs[0], sr)
    return str(output_path)


class ONNXRequestHandler(BaseHTTPRequestHandler):
    server_version = "CharacterTTS-ONNX/1.0"

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/":
            self._send_html(HTML_PAGE)
            return
        if path == "/api/characters":
            self._send_json({"characters": list_characters()})
            return
        self._send_json({"ok": False, "error": "Not found"}, status=404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path == "/api/characters":
            try:
                import cgi
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={
                        "REQUEST_METHOD": "POST",
                        "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                        "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
                    },
                )
                result = create_character(form)
                self._send_json({"ok": True, "character": result})
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
            return

        if path == "/api/generate":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                payload = json.loads(body.decode("utf-8"))
                character_id = str(payload.get("character_id", "")).strip()
                target_text = str(payload.get("target_text", "")).strip()
                language = str(payload.get("language", "English")).strip() or "English"
                output_path = generate_voice(character_id, target_text, language)
                relative = os.path.relpath(output_path, ROOT)
                self._send_json({
                    "ok": True,
                    "audio_url": f"/audio/{relative.replace(os.sep, '/')}",
                    "filename": Path(output_path).name,
                })
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
            return

        self._send_json({"ok": False, "error": "Not found"}, status=404)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html_text: str) -> None:
        body = html_text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(host: str = "0.0.0.0", port: int = 8001) -> None:
    print(f"Serving Character TTS Studio (ONNX path) on http://{host}:{port}")
    server = ThreadingHTTPServer((host, port), ONNXRequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down ONNX server...")
        server.server_close()


if __name__ == "__main__":
    run_server()
