from __future__ import annotations

import hashlib
import json
import logging
import re
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

logger = logging.getLogger("onnx_pipeline.onnx_runtime")


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
        self.tokenizer = self._load_tokenizer()
        self.embeddings = self._load_embeddings()
        self.model_config = self._load_model_config()
        self._load_sessions()

    def _load_model_config(self) -> dict[str, Any]:
        config_path = self.model_dir / "embeddings" / "config.json"
        if not config_path.exists():
            return {}
        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

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

    def _describe_session_shapes(self, session: Any) -> tuple[list[list[int | str]], list[list[int | str]]]:
        inputs = getattr(session, "get_inputs", lambda: [])()
        outputs = getattr(session, "get_outputs", lambda: [])()
        input_shapes = [list(getattr(inp, "shape", []) or []) for inp in inputs]
        output_shapes = [list(getattr(outp, "shape", []) or []) for outp in outputs]
        return input_shapes, output_shapes

    def _load_tokenizer(self) -> dict[str, Any]:
        tokenizer_dir = self.model_dir / "tokenizer"
        if not tokenizer_dir.exists():
            return {}

        vocab_path = tokenizer_dir / "vocab.json"
        merges_path = tokenizer_dir / "merges.txt"
        if not vocab_path.exists():
            return {}

        try:
            vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

        merges = []
        if merges_path.exists():
            try:
                merges = [line.strip().split() for line in merges_path.read_text(encoding="utf-8").splitlines() if line.strip() and "#" not in line]
            except Exception:
                merges = []

        reverse_vocab = {int(v): k for k, v in vocab.items()}
        return {"vocab": vocab, "reverse_vocab": reverse_vocab, "merges": merges}

    def _byte_encoder(self) -> dict[int, str]:
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        return {b: chr(c) for b, c in zip(bs, cs)}

    def _tokenize_text(self, text: str) -> list[int]:
        tokenizer = self.tokenizer
        if not tokenizer or not tokenizer.get("vocab"):
            token_count = max(1, len((text or "").split()) + 1)
            return list(range(token_count))[:1]

        vocab = tokenizer["vocab"]
        byte_encoder = self._byte_encoder()
        tokens: list[int] = []
        for piece in re.findall(r"'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+", text, flags=re.UNICODE):
            if not piece:
                continue
            encoded = "".join(byte_encoder[b] for b in piece.encode("utf-8"))
            if encoded in vocab:
                tokens.append(int(vocab[encoded]))
                continue
            tokens.extend(int(vocab.get(token, 0)) for token in [encoded] if token in vocab)
        if not tokens:
            tokens = [int(vocab.get("<|endoftext|>", 0) or 0)]
        return tokens

    def _load_embeddings(self) -> dict[str, Any]:
        embed_dir = self.model_dir / "embeddings"
        if not embed_dir.exists():
            return {}

        loaded: dict[str, Any] = {}
        for path in sorted(embed_dir.glob("*.npy")):
            name = path.stem
            try:
                loaded[name] = np.load(path)
            except Exception:
                continue

        speaker_path = embed_dir / "speaker_ids.json"
        if speaker_path.exists():
            try:
                loaded["speaker_ids"] = json.loads(speaker_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        config_path = embed_dir / "config.json"
        if config_path.exists():
            try:
                loaded["config"] = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        return loaded

    def _normalize_dim(self, dim: Any) -> int:
        if dim is None or dim == "":
            return 1
        if isinstance(dim, str):
            lowered = dim.lower()
            if lowered in {"batch", "b", "n", "sequence", "seq", "s", "t", "time", "token", "tokens"}:
                return 1
            try:
                return int(dim)
            except ValueError:
                return 1
        try:
            return int(dim)
        except (TypeError, ValueError):
            return 1

    def _project_to_cp_hidden(self, vector: np.ndarray) -> np.ndarray:
        if np is None:
            raise RuntimeError("NumPy is required for ORT inference.")

        projected = np.asarray(vector, dtype=np.float32)
        cp_proj_w = self.embeddings.get("cp_projection_weight")
        cp_proj_b = self.embeddings.get("cp_projection_bias")
        cp_hidden = int(self.model_config.get("code_predictor", {}).get("hidden_size", 1024))

        if cp_proj_w is not None and cp_proj_b is not None:
            w = np.asarray(cp_proj_w, dtype=np.float32)
            b = np.asarray(cp_proj_b, dtype=np.float32)
            if projected.ndim > 1:
                projected = projected.reshape(-1)
            if projected.size < w.shape[1]:
                pad = np.zeros((w.shape[1] - projected.size,), dtype=np.float32)
                projected = np.concatenate([projected.astype(np.float32), pad])
            if projected.size > w.shape[1]:
                projected = projected[: w.shape[1]]
            return (projected @ w.T) + b

        if projected.ndim > 1:
            projected = projected.reshape(-1)
        if projected.size < cp_hidden:
            pad = np.zeros((cp_hidden - projected.size,), dtype=np.float32)
            projected = np.concatenate([projected.astype(np.float32), pad])
        return projected[:cp_hidden].astype(np.float32)

    def _load_embedding_tables(self) -> dict[str, Any]:
        embed_dir = self.model_dir / "embeddings"
        if not embed_dir.exists():
            return {}

        tables: dict[str, Any] = {}
        names = [
            "text_embedding",
            "text_projection_fc1_weight",
            "text_projection_fc1_bias",
            "text_projection_fc2_weight",
            "text_projection_fc2_bias",
            "talker_codec_embedding",
            "cp_projection_weight",
            "cp_projection_bias",
        ]
        for name in names:
            path = embed_dir / f"{name}.npy"
            if path.exists():
                try:
                    tables[name] = np.load(path)
                except Exception:
                    tables[name] = None

        for idx in range(15):
            path = embed_dir / f"cp_codec_embedding_{idx}.npy"
            if path.exists():
                try:
                    tables[f"cp_codec_embedding_{idx}"] = np.load(path)
                except Exception:
                    tables[f"cp_codec_embedding_{idx}"] = None

        config_path = embed_dir / "config.json"
        if config_path.exists():
            try:
                tables["config"] = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                tables["config"] = {}

        return tables

    def _text_projection(self, vector: np.ndarray, embeddings: dict[str, Any]) -> np.ndarray:
        if np is None:
            raise RuntimeError("NumPy is required for ORT inference.")

        fc1_w = embeddings.get("text_projection_fc1_weight")
        fc1_b = embeddings.get("text_projection_fc1_bias")
        fc2_w = embeddings.get("text_projection_fc2_weight")
        fc2_b = embeddings.get("text_projection_fc2_bias")
        if fc1_w is None or fc1_b is None or fc2_w is None or fc2_b is None:
            return vector.astype(np.float32)

        hidden = np.matmul(vector.astype(np.float32), fc1_w.T) + fc1_b
        hidden = hidden / (1.0 + np.exp(-hidden))
        out = np.matmul(hidden, fc2_w.T) + fc2_b
        return out.astype(np.float32)

    def _tokenize_text(self, text: str) -> list[int]:
        if not text:
            return []
        if self.tokenizer and self.tokenizer.get("vocab"):
            vocab = self.tokenizer["vocab"]
            if "" in vocab:
                return [int(vocab[""])]
            token_ids = []
            for ch in text:
                token = vocab.get(ch)
                if token is not None:
                    token_ids.append(int(token))
                else:
                    token_ids.append(0)
            if token_ids:
                return token_ids
        return [max(1, len(text.strip().split()))]

    def _lookup_talker_codec_embedding(self, token_id: int, embeddings: dict[str, Any]) -> np.ndarray:
        table = embeddings.get("talker_codec_embedding")
        if table is None or token_id < 0 or token_id >= table.shape[0]:
            return np.zeros((table.shape[1],), dtype=np.float32) if table is not None else np.zeros((1,), dtype=np.float32)
        return np.asarray(table[token_id], dtype=np.float32)

    def _lookup_cp_codec_embedding(self, group_index: int, token_id: int, embeddings: dict[str, Any]) -> np.ndarray:
        key = f"cp_codec_embedding_{group_index}"
        table = embeddings.get(key)
        if table is None or token_id < 0 or token_id >= table.shape[0]:
            return np.zeros((table.shape[1],), dtype=np.float32) if table is not None else np.zeros((1,), dtype=np.float32)
        return np.asarray(table[token_id], dtype=np.float32)

    def _build_prefill_inputs(self, text: str, embeddings: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        token_ids = self._tokenize_text(text)
        if not token_ids:
            token_ids = [0]

        token_embeddings = []
        for token_id in token_ids:
            token_vec = np.asarray(self._lookup_talker_codec_embedding(token_id, embeddings), dtype=np.float32)
            token_embeddings.append(token_vec)
        if not token_embeddings:
            token_embeddings = [np.zeros((1,), dtype=np.float32)]
        token_embedding_matrix = np.stack(token_embeddings, axis=0).astype(np.float32)

        text_embedding = embeddings.get("text_embedding")
        if text_embedding is not None and token_ids:
            proj = self._text_projection(text_embedding[token_ids[0]], embeddings) if len(token_ids) == 1 else np.stack([
                self._text_projection(text_embedding[token_id], embeddings) for token_id in token_ids
            ], axis=0)
            if token_embedding_matrix.shape[0] == proj.shape[0]:
                token_embedding_matrix = proj.astype(np.float32)

        sequence_len = token_embedding_matrix.shape[0]
        inputs_embeds = token_embedding_matrix[np.newaxis, :, :].astype(np.float32)
        attention_mask = np.ones((1, sequence_len), dtype=np.int64)
        position_ids = np.arange(sequence_len, dtype=np.int64)[np.newaxis, :].repeat(3, axis=0)
        return inputs_embeds, attention_mask, position_ids

    def _sample_token_from_logits(self, logits: np.ndarray) -> int:
        logits = np.asarray(logits, dtype=np.float32)
        flat = logits.reshape(-1)
        if flat.size == 0:
            return 0
        return int(np.argmax(flat))

    def _sample_code_predictor_token(self, logits: np.ndarray) -> int:
        logits = np.asarray(logits, dtype=np.float32).reshape(-1)
        codebook_size = int(self.model_config.get("code_predictor", {}).get("vocab_size", 2048))
        if logits.size == 0 or codebook_size <= 0:
            return 0
        return int(np.argmax(logits[:codebook_size]))

    def _sample_talker_codec_token(self, logits: np.ndarray, allow_eos: bool) -> int:
        logits = np.asarray(logits, dtype=np.float32).reshape(-1)
        codebook_size = 2048
        if logits.size == 0:
            return 0

        masked_logits = np.full_like(logits, -np.inf)
        masked_logits[:codebook_size] = logits[:codebook_size]
        if allow_eos:
            eos_token_id = int(self.model_config.get("talker", {}).get("codec_eos_token_id", 2150))
            if 0 <= eos_token_id < logits.size:
                masked_logits[eos_token_id] = logits[eos_token_id]
        return int(np.argmax(masked_logits))

    def _run_talker_prefill(self, inputs_embeds: np.ndarray, attention_mask: np.ndarray, position_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        session = self.sessions.get("talker_prefill") or self.sessions.get("talker")
        if session is None:
            raise RuntimeError("No talker prefill session is available.")
        feed = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        outputs = session.run(None, feed)
        logits = np.asarray(outputs[0])
        hidden_states = np.asarray(outputs[1])
        num_layers = int(self.model_config.get("talker", {}).get("num_hidden_layers", 28))
        past_keys = np.stack([
            np.asarray(outputs[2 + layer * 2]) for layer in range(num_layers)
        ], axis=0)
        past_values = np.stack([
            np.asarray(outputs[3 + layer * 2]) for layer in range(num_layers)
        ], axis=0)
        return logits, hidden_states, past_keys, past_values

    def _run_talker_decode(self, inputs_embeds: np.ndarray, attention_mask: np.ndarray, position_ids: np.ndarray, past_keys: np.ndarray, past_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        session = self.sessions.get("talker_decode") or self.sessions.get("talker")
        if session is None:
            raise RuntimeError("No talker decode session is available.")
        feed = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_keys": past_keys,
            "past_values": past_values,
        }
        outputs = session.run(None, feed)
        logits = np.asarray(outputs[0])
        hidden_states = np.asarray(outputs[1])
        present_keys = np.asarray(outputs[2])
        present_values = np.asarray(outputs[3])
        return logits, hidden_states, present_keys, present_values

    def _run_code_predictor(self, inputs_embeds: np.ndarray, generation_step: int, past_keys: list[np.ndarray], past_values: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        session = self.sessions.get("code_predictor")
        if session is None:
            raise RuntimeError("No code predictor session is available.")
        feed = {
            "inputs_embeds": inputs_embeds,
            "generation_steps": np.asarray([generation_step], dtype=np.int64),
            "past_keys": np.stack(past_keys, axis=0) if past_keys else np.zeros((1, 1, 1, 1, 1), dtype=np.float32),
            "past_values": np.stack(past_values, axis=0) if past_values else np.zeros((1, 1, 1, 1, 1), dtype=np.float32),
        }
        outputs = session.run(None, feed)
        logits = np.asarray(outputs[0])
        present_keys = np.asarray(outputs[1])
        present_values = np.asarray(outputs[2])
        return logits, present_keys, present_values

    def _build_onnx_input(self, input_meta: Any, *, text: str, ref_text: str) -> np.ndarray:
        if np is None:
            raise RuntimeError("NumPy is required for ORT inference.")

        name = (getattr(input_meta, "name", "") or "").lower()
        shape = list(getattr(input_meta, "shape", []) or [])
        dims = [self._normalize_dim(dim) for dim in shape]

        if "generation_steps" in name or "generation_step" in name or "step" in name:
            dtype = np.int64 if "int64" in str(getattr(input_meta, "type", "")).lower() else np.int32
            return np.array([1], dtype=dtype)

        if "past_keys" in name or "past_values" in name or "key_cache" in name or "value_cache" in name:
            if not dims:
                dims = [1, 1, 1, 1]
            while len(dims) < 4:
                dims.append(1)
            return np.zeros(tuple(dims), dtype=np.float32)

        if "input_ids" in name or "token_ids" in name or "prompt_ids" in name or "tokens" in name:
            tokens = self._tokenize_text(text or ref_text or "")
            token_ids = np.array(tokens[: max(1, len(tokens))], dtype=np.int64)
            if not dims:
                dims = [1, len(token_ids)]
            elif len(dims) == 1:
                dims = [len(token_ids)]
            else:
                dims[0] = 1
                if len(dims) >= 2 and dims[1] <= 0:
                    dims[1] = len(token_ids)
            return np.reshape(token_ids[: max(1, dims[1])], tuple(dims)).astype(np.int64)

        if "inputs_embeds" in name or "embeds" in name:
            cp_hidden = int(self.model_config.get("code_predictor", {}).get("hidden_size", 1024))
            if self.embeddings and self.model_config.get("code_predictor"):
                shape_from_model = [1, 2, cp_hidden]
                hidden = np.zeros(shape_from_model, dtype=np.float32)

                text_embedding = self.embeddings.get("text_embedding")
                talker_embedding = self.embeddings.get("talker_codec_embedding")
                if text_embedding is not None:
                    base = np.asarray(text_embedding[0], dtype=np.float32)
                    hidden[0, 0] = self._project_to_cp_hidden(base)
                if talker_embedding is not None:
                    cp_token = np.asarray(talker_embedding[0], dtype=np.float32)
                    hidden[0, 1] = self._project_to_cp_hidden(cp_token)
                return hidden.astype(np.float32)

            tokens = self._tokenize_text(text or ref_text or "")
            token_ids = np.array(tokens, dtype=np.int64)
            if self.embeddings and "text_embedding" in self.embeddings:
                text_embedding = self.embeddings["text_embedding"]
                if token_ids.size:
                    embedded = text_embedding[token_ids]
                    embedded = np.asarray(embedded, dtype=np.float32)
                    if embedded.ndim == 1:
                        embedded = embedded[np.newaxis, :]
                    return embedded[np.newaxis, :, :].astype(np.float32)
            token_count = max(1, len(token_ids))
            if not dims:
                dims = [1, token_count]
            elif len(dims) == 1:
                dims = [1, token_count]
            else:
                dims[0] = 1
                if len(dims) >= 2 and dims[1] <= 0:
                    dims[1] = token_count
            return np.zeros(tuple(dims), dtype=np.float32)

        if "attention_mask" in name or "mask" in name:
            token_count = max(1, len(self._tokenize_text(text or ref_text or "")))
            if not dims:
                dims = [1, token_count]
            else:
                dims[0] = 1
                if len(dims) >= 2 and dims[1] <= 0:
                    dims[1] = token_count
            return np.ones(tuple(dims), dtype=np.int64)

        if "position" in name:
            token_count = max(1, len(self._tokenize_text(text or ref_text or "")))
            positional = np.arange(token_count, dtype=np.int64)
            if not dims:
                return positional.reshape(1, -1)
            if len(dims) == 1:
                return positional.astype(np.int64)
            dims[0] = 1
            if len(dims) >= 2 and dims[1] <= 0:
                dims[1] = token_count
            return np.broadcast_to(positional.reshape(1, -1), tuple(dims)).astype(np.int64)

        if not dims:
            dims = [1]
        return np.zeros(tuple(dims), dtype=np.float32 if "float" in str(getattr(input_meta, "type", "")).lower() else np.int64)

    def _load_sessions(self) -> None:
        if ort is None:
            return
        candidate_paths = {
            "code_predictor": Path(self.bundle.encoder_path) if self.bundle.encoder_path else None,
            "talker_prefill": Path(self.model_dir / "talker_prefill.onnx") if (self.model_dir / "talker_prefill.onnx").exists() else (Path(self.bundle.decoder_path) if self.bundle.decoder_path and Path(self.bundle.decoder_path).name == "talker_prefill.onnx" else None),
            "talker_decode": Path(self.model_dir / "talker_decode.onnx") if (self.model_dir / "talker_decode.onnx").exists() else None,
            "talker": Path(self.model_dir / "talker_prefill.onnx") if (self.model_dir / "talker_prefill.onnx").exists() else (Path(self.bundle.decoder_path) if self.bundle.decoder_path else None),
            "vocoder": Path(self.model_dir / "vocoder.onnx") if (self.model_dir / "vocoder.onnx").exists() else None,
        }
        for name, model_path in candidate_paths.items():
            if model_path is None or not model_path.exists():
                continue
            try:
                session = ort.InferenceSession(str(model_path), providers=[self.provider])
                self.sessions[name] = session
                input_shapes, output_shapes = self._describe_session_shapes(session)
                logger.info("Loaded ONNX model %s from %s | input_shapes=%s output_shapes=%s", name, model_path, input_shapes, output_shapes)
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
                if np is None:
                    raise RuntimeError("NumPy is required for ORT inference.")

                prefill_session = self.sessions.get("talker_prefill") or self.sessions.get("talker")
                decode_session = self.sessions.get("talker_decode") or self.sessions.get("talker")
                cp_session = self.sessions.get("code_predictor")
                vocoder_session = self.sessions.get("vocoder")
                if prefill_session is None:
                    raise RuntimeError("Talker prefill ONNX session is not available.")
                if decode_session is None:
                    raise RuntimeError("Talker decode ONNX session is not available.")
                if cp_session is None:
                    raise RuntimeError("Code predictor ONNX session is not available.")
                if vocoder_session is None:
                    logger.warning("No vocoder session found; falling back to synthetic waveform output.")
                    return self._synthetic_waveform(text=text, ref_audio=ref_audio, ref_text=ref_text, language=language)

                text_embedding = self.embeddings.get("text_embedding")
                if text_embedding is None:
                    raise RuntimeError("The ONNX bundle is missing text_embedding.npy.")

                token_ids = self._tokenize_text(text)
                if not token_ids:
                    token_ids = [0]

                text_tokens = np.asarray(token_ids, dtype=np.int64)
                text_embeds = np.asarray(text_embedding[text_tokens], dtype=np.float32)
                if text_embeds.ndim == 1:
                    text_embeds = text_embeds[np.newaxis, :]
                text_proj = np.stack([
                    self._text_projection(text_embeds[i], self.embeddings)
                    for i in range(text_embeds.shape[0])
                ], axis=0).astype(np.float32)

                inputs_embeds = np.asarray(text_proj, dtype=np.float32)[np.newaxis, :, :]
                attention_mask = np.ones((1, inputs_embeds.shape[1]), dtype=np.int64)
                position_ids = np.broadcast_to(np.arange(inputs_embeds.shape[1], dtype=np.int64)[None, None, :], (3, 1, inputs_embeds.shape[1])).copy()

                logger.info(
                    "Running ONNX inference model=%s input_shapes=%s",
                    "talker_prefill",
                    [list(inputs_embeds.shape), list(attention_mask.shape), list(position_ids.shape)],
                )
                prefill_outputs = prefill_session.run(None, {
                    "inputs_embeds": inputs_embeds,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                })
                logits = np.asarray(prefill_outputs[0])
                hidden_states = np.asarray(prefill_outputs[1])
                num_layers = int(self.model_config.get("talker", {}).get("num_hidden_layers", 28))
                past_keys = np.stack([
                    np.asarray(prefill_outputs[2 + layer * 2]) for layer in range(num_layers)
                ], axis=0)
                past_values = np.stack([
                    np.asarray(prefill_outputs[3 + layer * 2]) for layer in range(num_layers)
                ], axis=0)

                generated_codes = []
                generated_tokens = []
                cp_total_layers = int(self.model_config.get("code_predictor", {}).get("num_hidden_layers", 5))
                if cp_total_layers <= 0:
                    cp_total_layers = 5
                num_kv_heads = int(self.model_config.get("code_predictor", {}).get("num_key_value_heads", 8))
                if num_kv_heads <= 0:
                    num_kv_heads = 8
                head_dim = int(self.model_config.get("code_predictor", {}).get("head_dim", 128))
                if head_dim <= 0:
                    head_dim = 128

                for step in range(8):
                    last_logits = logits[0, -1, :] if logits.ndim == 3 else logits.reshape(-1)
                    group0_token = self._sample_talker_codec_token(last_logits, allow_eos=step >= 2)
                    if group0_token == int(self.model_config.get("talker", {}).get("codec_eos_token_id", 2150)):
                        break
                    generated_tokens.append(group0_token)

                    codes = [group0_token]
                    last_hidden = np.asarray(hidden_states[0, -1, :], dtype=np.float32)
                    projected_hidden = self._project_to_cp_hidden(last_hidden)
                    group0_embed = np.asarray(self.embeddings["talker_codec_embedding"][group0_token], dtype=np.float32)
                    projected_group0 = self._project_to_cp_hidden(group0_embed)
                    cp_inputs = np.stack([projected_hidden, projected_group0], axis=0)[np.newaxis, :, :].astype(np.float32)

                    cp_past_keys = np.zeros((cp_total_layers, 1, num_kv_heads, 1, head_dim), dtype=np.float32)
                    cp_past_values = np.zeros_like(cp_past_keys)
                    for group_idx in range(1, 16):
                        seq_len = 2 if group_idx == 1 else 1
                        cp_feed = {
                            "inputs_embeds": cp_inputs[:, -seq_len:, :],
                            "generation_steps": np.asarray([group_idx - 1], dtype=np.int64),
                            "past_keys": cp_past_keys,
                            "past_values": cp_past_values,
                        }
                        cp_outputs = cp_session.run(None, cp_feed)
                        cp_logits = np.asarray(cp_outputs[0])
                        next_token = self._sample_code_predictor_token(cp_logits[0, -1, :])
                        codes.append(next_token)
                        cp_past_keys = np.asarray(cp_outputs[1])
                        cp_past_values = np.asarray(cp_outputs[2])
                        if group_idx < 15:
                            key = f"cp_codec_embedding_{group_idx - 1}"
                            next_embed = self.embeddings.get(key)
                            if next_embed is None:
                                next_embed = self.embeddings.get("cp_codec_embedding_0")
                            if next_embed is None:
                                next_embed = self.embeddings.get("talker_codec_embedding")
                            next_vec = np.asarray(next_embed[next_token], dtype=np.float32)
                            projected = self._project_to_cp_hidden(next_vec)
                            cp_inputs = projected[np.newaxis, np.newaxis, :].astype(np.float32)

                    generated_codes.append(codes)
                    next_input = np.asarray(self.embeddings["talker_codec_embedding"][codes[0]], dtype=np.float32).copy()
                    for group_idx in range(1, 16):
                        key = f"cp_codec_embedding_{group_idx - 1}"
                        emb = self.embeddings.get(key)
                        if emb is None:
                            emb = self.embeddings.get("talker_codec_embedding")
                        next_input += np.asarray(emb[codes[group_idx]], dtype=np.float32)
                    if step < len(text_tokens):
                        next_input += np.asarray(text_proj[step], dtype=np.float32)
                    else:
                        next_input += np.zeros_like(next_input)

                    prefill_len = inputs_embeds.shape[1]
                    new_len = prefill_len + step + 1
                    attention_mask_decode = np.ones((1, new_len), dtype=np.int64)
                    position_ids_decode = np.full((3, 1, 1), prefill_len + step, dtype=np.int64)
                    decode_feed = {
                        "inputs_embeds": next_input[np.newaxis, np.newaxis, :].astype(np.float32),
                        "attention_mask": attention_mask_decode,
                        "position_ids": position_ids_decode,
                        "past_keys": past_keys,
                        "past_values": past_values,
                    }
                    decode_outputs = decode_session.run(None, decode_feed)
                    logits = np.asarray(decode_outputs[0])
                    hidden_states = np.asarray(decode_outputs[1])
                    past_keys = np.asarray(decode_outputs[2])
                    past_values = np.asarray(decode_outputs[3])

                if not generated_codes:
                    generated_codes = [[0] * 16]

                code_matrix = np.asarray(generated_codes, dtype=np.int64)
                if code_matrix.shape[1] != 16:
                    code_matrix = np.pad(code_matrix, ((0, 0), (0, 16 - code_matrix.shape[1])), constant_values=0)
                code_tensor = code_matrix.T[np.newaxis, :, :]
                logger.info("Running vocoder model=%s codes_shape=%s", "vocoder", list(code_tensor.shape))
                waveform_out = vocoder_session.run(None, {"codes": code_tensor})[0]
                waveform = np.asarray(waveform_out).reshape(-1).astype(np.float32)
                return [waveform], 24000
            except Exception as exc:
                logger.exception(
                    "ONNX inference failed for this model export. The exported graph is not compatible with the current generate_voice_clone implementation."
                )
                raise RuntimeError(
                    "The ONNX model bundle is not compatible with this runtime implementation. "
                    "Use the PyTorch server or export a matching ONNX graph."
                ) from exc

        logger.warning("No usable ONNX session found; falling back to synthetic waveform output.")
        return self._synthetic_waveform(text=text, ref_audio=ref_audio, ref_text=ref_text, language=language)
