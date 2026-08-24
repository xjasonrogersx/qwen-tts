#!/usr/bin/env python3
from __future__ import annotations

import cgi
import html
import importlib.util
import json
import mimetypes
import os
import re
import shutil
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import soundfile as sf

MODULE_PATH = SRC_DIR / "tts" / "__init__.py"
SPEC = importlib.util.spec_from_file_location("tts_pkg", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Unable to load TTS package from {MODULE_PATH}")
TTS_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TTS_MODULE)
validate_audio_file = TTS_MODULE.validate_audio_file

try:
    from qwen_tts import Qwen3TTSModel
except Exception:  # pragma: no cover
    Qwen3TTSModel = None

CHARACTERS_DIR = ROOT / "characters"
CHARACTERS_DIR.mkdir(exist_ok=True)
GENERATED_DIR = ROOT / "generated"
GENERATED_DIR.mkdir(exist_ok=True)

MODEL = None

HTML_PAGE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Character TTS Studio</title>
    <style>
      :root {
        --bg: #0f172a;
        --panel: #111827;
        --panel-2: #1f2937;
        --border: #334155;
        --text: #e2e8f0;
        --muted: #94a3b8;
        --accent: #7c3aed;
        --accent-2: #22c55e;
        --danger: #ef4444;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        background: linear-gradient(180deg, #020817 0%, #0f172a 100%);
        color: var(--text);
        font-family: Arial, sans-serif;
      }
      .wrap {
        max-width: 1200px;
        margin: 0 auto;
        padding: 32px 20px 64px;
      }
      .grid {
        display: grid;
        grid-template-columns: 1.1fr 1.6fr;
        gap: 24px;
      }
      .card {
        background: rgba(17, 24, 39, 0.9);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
      }
      h1, h2, h3 { margin-top: 0; }
      label { display: block; margin: 12px 0 6px; color: var(--muted); }
      input, textarea, select, button {
        width: 100%;
        border-radius: 10px;
        border: 1px solid var(--border);
        background: #0b1220;
        color: var(--text);
        padding: 10px 12px;
        font-size: 14px;
      }
      textarea { min-height: 110px; resize: vertical; }
      button {
        background: linear-gradient(135deg, var(--accent), #4f46e5);
        border: none;
        font-weight: 700;
        cursor: pointer;
        margin-top: 10px;
      }
      button.secondary {
        background: linear-gradient(135deg, var(--accent-2), #16a34a);
      }
      .row {
        display: flex;
        gap: 12px;
        align-items: center;
      }
      .character-list {
        display: grid;
        gap: 10px;
      }
      .character-item {
        display: flex;
        gap: 12px;
        align-items: center;
        background: var(--panel-2);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 10px 12px;
      }
      .avatar {
        width: 52px;
        height: 52px;
        border-radius: 50%;
        object-fit: cover;
        background: #0b1220;
        border: 1px solid var(--border);
      }
      .muted { color: var(--muted); }
      .status {
        margin-top: 12px;
        padding: 10px 12px;
        border-radius: 10px;
        background: rgba(15, 118, 110, 0.15);
        border: 1px solid rgba(45, 212, 191, 0.4);
        display: none;
      }
      .status.error {
        background: rgba(239, 68, 68, 0.1);
        border-color: rgba(239, 68, 68, 0.35);
      }
      .result-box {
        margin-top: 16px;
        display: none;
      }
      audio {
        width: 100%;
        margin-top: 12px;
      }
      .hint {
        font-size: 12px;
        color: var(--muted);
      }
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="grid">
        <div class="card">
          <h2>Create Character</h2>
          <form id="new-character-form" enctype="multipart/form-data">
            <label>Name</label>
            <input name="name" required placeholder="e.g. Ava Narrator" />

            <label>Description</label>
            <input name="description" placeholder="Optional flavor text" />

            <label>Reference audio (.wav, .flac, .mp3, .ogg)</label>
            <input type="file" name="ref_audio" accept="audio/*" required />

            <label>Reference text</label>
            <textarea name="ref_text" required placeholder="Paste the transcript for the reference clip..."></textarea>

            <label>Avatar image (optional)</label>
            <input type="file" name="avatar" accept="image/*" />

            <button type="submit">Save character</button>
          </form>
          <div id="character-status" class="status"></div>

          <h2 style="margin-top: 28px;">Saved Characters</h2>
          <div id="character-list" class="character-list"></div>
        </div>

        <div class="card">
          <h2>Generate Audio</h2>
          <form id="generate-form">
            <label>Select Character</label>
            <select id="character-select" name="character_id" required></select>

            <label>Target Text</label>
            <textarea id="target-text" name="target_text" required placeholder="Write what you want the selected character to say..."></textarea>

            <label>Language</label>
            <input name="language" value="English" placeholder="English" />

            <button class="secondary" type="submit">Generate speech</button>
          </form>

          <div id="generate-status" class="status"></div>

          <div id="result-box" class="result-box">
            <h3>Output</h3>
            <audio id="generated-audio" controls></audio>
            <a id="download-link" href="#" download>Download WAV</a>
          </div>
        </div>
      </div>
    </div>

    <script>
      const charList = document.getElementById('character-list');
      const charSelect = document.getElementById('character-select');
      const statusBox = document.getElementById('character-status');
      const generateStatus = document.getElementById('generate-status');
      const resultBox = document.getElementById('result-box');
      const audioEl = document.getElementById('generated-audio');
      const downloadLink = document.getElementById('download-link');

      function showStatus(el, text, isError = false) {
        el.textContent = text;
        el.style.display = 'block';
        el.classList.toggle('error', isError);
      }

      async function refreshCharacters() {
        const res = await fetch('/api/characters');
        const data = await res.json();
        const chars = data.characters || [];

        charList.innerHTML = '';
        charSelect.innerHTML = '<option value="">Select a character</option>';

        for (const ch of chars) {
          const row = document.createElement('div');
          row.className = 'character-item';
          const avatar = ch.avatar ? `<img class="avatar" src="${ch.avatar}" alt="${ch.name}" />` : `<div class="avatar" style="display:flex;align-items:center;justify-content:center;">${ch.name.slice(0,1).toUpperCase()}</div>`;
          row.innerHTML = `
            ${avatar}
            <div style="flex:1;">
              <div><strong>${ch.name}</strong></div>
              <div class="muted">${ch.description || 'No description'}</div>
            </div>
          `;
          row.addEventListener('click', () => {
            charSelect.value = ch.id;
          });
          charList.appendChild(row);

          const option = document.createElement('option');
          option.value = ch.id;
          option.textContent = ch.name;
          charSelect.appendChild(option);
        }
      }

      document.getElementById('new-character-form').addEventListener('submit', async (event) => {
        event.preventDefault();
        const formData = new FormData(event.target);
        showStatus(statusBox, 'Saving character...', false);

        const res = await fetch('/api/characters', {
          method: 'POST',
          body: formData
        });
        const data = await res.json();

        if (!res.ok || !data.ok) {
          showStatus(statusBox, data.error || 'Failed to save character', true);
          return;
        }

        showStatus(statusBox, `Saved ${data.character.name}`);
        event.target.reset();
        await refreshCharacters();
      });

      document.getElementById('generate-form').addEventListener('submit', async (event) => {
        event.preventDefault();
        const formData = new FormData(event.target);
        const characterId = formData.get('character_id');
        const targetText = formData.get('target_text');
        const language = formData.get('language') || 'English';

        if (!characterId || !targetText.trim()) {
          showStatus(generateStatus, 'Choose a character and add target text.', true);
          return;
        }

        showStatus(generateStatus, 'Generating speech...', false);
        resultBox.style.display = 'none';

        const res = await fetch('/api/generate', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ character_id: characterId, target_text: targetText, language })
        });

        const data = await res.json();
        if (!res.ok || !data.ok) {
          showStatus(generateStatus, data.error || 'Generation failed', true);
          return;
        }

        audioEl.src = data.audio_url;
        downloadLink.href = data.audio_url;
        downloadLink.setAttribute('download', data.filename || 'voice.wav');
        resultBox.style.display = 'block';
        showStatus(generateStatus, 'Audio generated successfully.');
      });

      refreshCharacters();
    </script>
  </body>
</html>
"""


def slugify(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or f"character-{int(time.time() * 1000)}"


def sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("._")
    return cleaned or "upload"


def ensure_character_dir(name: str) -> Path:
    slug = slugify(name)
    target_dir = CHARACTERS_DIR / slug
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def save_uploaded_file(file_field, destination: Path) -> str:
    if hasattr(file_field, "filename") and file_field.filename:
        filename = sanitize_filename(file_field.filename)
        destination = destination.with_name(filename)
        with open(destination, "wb") as out:
            shutil.copyfileobj(file_field.file, out)
        return destination.name
    raise ValueError("No upload was received.")


def list_characters() -> list[dict]:
    items: list[dict] = []
    if not CHARACTERS_DIR.exists():
        return items
    for path in sorted(CHARACTERS_DIR.iterdir()):
        if not path.is_dir():
            continue
        metadata_path = path / "metadata.json"
        if not metadata_path.exists():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        avatar_path = path / metadata.get("avatar", "")
        avatar_url = f"/audio/characters/{path.name}/{metadata.get('avatar', '')}" if metadata.get('avatar') and avatar_path.exists() else ""
        items.append({
            "id": path.name,
            "name": metadata.get("name", path.name),
            "description": metadata.get("description", ""),
            "ref_text": metadata.get("ref_text", ""),
            "avatar": avatar_url,
            "ref_audio": f"/audio/characters/{path.name}/{metadata.get('ref_audio', '')}" if metadata.get('ref_audio') else "",
        })
    return items


def load_character(identifier: str) -> tuple[Path, dict]:
    target_dir = CHARACTERS_DIR / identifier
    metadata_path = target_dir / "metadata.json"
    if not target_dir.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Character not found: {identifier}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return target_dir, metadata


def create_character(form) -> dict:
    name = (form.getfirst("name") or "").strip()
    description = (form.getfirst("description") or "").strip()
    ref_text = (form.getfirst("ref_text") or "").strip()
    if not name:
        raise ValueError("Character name is required.")
    if not ref_text:
        raise ValueError("Reference text is required.")

    ref_audio_item = form.getfirst("ref_audio")
    if not ref_audio_item:
        raise ValueError("Reference audio is required.")

    file_field = None
    for item in form.list:
        if getattr(item, "name", None) == "ref_audio" and getattr(item, "filename", None):
            file_field = item
            break
    if file_field is None:
        raise ValueError("Reference audio file was not uploaded.")

    char_dir = ensure_character_dir(name)
    char_dir.mkdir(parents=True, exist_ok=True)

    ref_audio_name = save_uploaded_file(file_field, char_dir / "ref_audio")
    ref_audio_path = char_dir / ref_audio_name

    ok, reason = validate_audio_file(ref_audio_path)
    if not ok:
        if ref_audio_path.exists():
            ref_audio_path.unlink(missing_ok=True)
        raise ValueError(f"Reference audio is invalid: {reason}")

    avatar_name = ""
    avatar_field = None
    for item in form.list:
        if getattr(item, "name", None) == "avatar" and getattr(item, "filename", None):
            avatar_field = item
            break
    if avatar_field is not None:
        avatar_name = save_uploaded_file(avatar_field, char_dir / "avatar")

    metadata = {
        "name": name,
        "description": description,
        "ref_text": ref_text,
        "ref_audio": ref_audio_name,
        "avatar": avatar_name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (char_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {"id": char_dir.name, "name": name, "description": description, "ref_text": ref_text, "avatar": f"/audio/characters/{char_dir.name}/{avatar_name}" if avatar_name else "", "ref_audio": f"/audio/characters/{char_dir.name}/{ref_audio_name}"}


def get_model():
    global MODEL
    if MODEL is not None:
        return MODEL
    if Qwen3TTSModel is None:
        raise RuntimeError("qwen_tts is not installed in this environment.")
    MODEL = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        device_map="cpu",
    )
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

    model = get_model()
    wavs, sr = model.generate_voice_clone(
        text=text,
        language=language,
        ref_audio=str(ref_audio),
        ref_text=metadata.get("ref_text", ""),
    )
    output_name = f"generated_{int(time.time() * 1000)}.wav"
    output_path = GENERATED_DIR / output_name
    sf.write(output_path, wavs[0], sr)
    return str(output_path)


class TTSRequestHandler(BaseHTTPRequestHandler):
    server_version = "CharacterTTS/1.0"

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path == "/":
            self._send_html(HTML_PAGE)
            return

        if path == "/api/characters":
            self._send_json({"characters": list_characters()})
            return

        if path.startswith("/audio/"):
            self._serve_file(path)
            return

        self._send_json({"ok": False, "error": "Not found"}, status=404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path == "/api/characters":
            try:
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

    def _serve_file(self, url_path: str) -> None:
        rel = url_path.removeprefix("/audio/")
        candidates = [
            rel,
            rel if rel.startswith("characters/") else f"characters/{rel}",
            rel if rel.startswith("characters/") else f"characters/{rel.lstrip('./')}",
            rel.removeprefix("generated/") if rel.startswith("generated/") else f"generated/{rel}",
            rel.removeprefix("generated/") if rel.startswith("generated/") else f"generated/{rel.lstrip('./')}",
        ]

        file_path = None
        for candidate in candidates:
            candidate_path = (ROOT / unquote(candidate)).resolve()
            if candidate_path.exists() and (ROOT in candidate_path.parents or candidate_path == ROOT):
                file_path = candidate_path
                break

        if file_path is None:
            self._send_json({"ok": False, "error": "File not found"}, status=404)
            return

        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type is None:
            mime_type = "application/octet-stream"

        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    print(f"Serving Character TTS Studio on http://{host}:{port}")
    server = ThreadingHTTPServer((host, port), TTSRequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()


if __name__ == "__main__":
    run_server()
