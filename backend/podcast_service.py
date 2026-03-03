"""Podcast generation service using notebook content from chunks table."""

import io
import os
import re
import textwrap
import tempfile
import wave
import aifc
import audioop
import subprocess
import shutil
import numpy as np
from datetime import datetime
from pathlib import Path

from huggingface_hub import InferenceClient
import requests
import pyttsx3

try:
    import torch
    from transformers import AutoProcessor, AutoModelForTextToSpectrogram, SpeechT5HifiGan
except Exception:
    torch = None
    AutoProcessor = None
    AutoModelForTextToSpectrogram = None
    SpeechT5HifiGan = None

from backend.artifacts_service import create_artifact
from backend.db import supabase
from backend.storage import get_artifacts_path, save_file

HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_MODEL = os.getenv("HF_PODCAST_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
HF_TTS_MODEL = os.getenv("HF_TTS_MODEL", "microsoft/speecht5_tts")
HF_TTS_FALLBACK_MODELS = [
    "microsoft/speecht5_tts",
    "facebook/mms-tts",
    "hexgrad/Kokoro-82M",
    "facebook/mms-tts-eng",
    "suno/bark-small",
]

_SPEECHT5_PROCESSOR = None
_SPEECHT5_MODEL = None
_SPEECHT5_VOCODER = None


def _tts_models() -> list[str]:
    candidates = [HF_TTS_MODEL] + HF_TTS_FALLBACK_MODELS
    seen = set()
    ordered = []
    for model in candidates:
        if model and model not in seen:
            seen.add(model)
            ordered.append(model)
    return ordered


def _load_speecht5():
    global _SPEECHT5_PROCESSOR, _SPEECHT5_MODEL, _SPEECHT5_VOCODER
    if AutoProcessor is None or AutoModelForTextToSpectrogram is None or SpeechT5HifiGan is None or torch is None:
        raise ValueError("transformers/torch SpeechT5 dependencies are not available")

    if _SPEECHT5_PROCESSOR is None:
        _SPEECHT5_PROCESSOR = AutoProcessor.from_pretrained("microsoft/speecht5_tts")
    if _SPEECHT5_MODEL is None:
        _SPEECHT5_MODEL = AutoModelForTextToSpectrogram.from_pretrained("microsoft/speecht5_tts")
    if _SPEECHT5_VOCODER is None:
        _SPEECHT5_VOCODER = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    _SPEECHT5_MODEL.eval()
    _SPEECHT5_VOCODER.eval()


def _wav_from_float32(samples: np.ndarray, sample_rate: int = 16000) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())
    return buffer.getvalue()


def _synthesize_local_tts_with_transformers(text: str) -> bytes:
    _load_speecht5()
    chunks = textwrap.wrap(text, width=350, break_long_words=False, break_on_hyphens=False)
    if not chunks:
        raise ValueError("No text provided for SpeechT5 generation")

    speaker_embeddings = torch.zeros((1, 512), dtype=torch.float32)
    all_parts = []

    with torch.inference_mode():
        for chunk in chunks:
            inputs = _SPEECHT5_PROCESSOR(text=chunk, return_tensors="pt")
            speech = _SPEECHT5_MODEL.generate_speech(
                inputs["input_ids"],
                speaker_embeddings,
                vocoder=_SPEECHT5_VOCODER,
            )
            samples = speech.cpu().numpy().astype(np.float32)
            all_parts.append(_wav_from_float32(samples, sample_rate=16000))

    return _concat_wav_bytes(all_parts)


def _get_notebook_chunks(notebook_id: str, limit: int = 300) -> list[dict]:
    result = (
        supabase.table("chunks")
        .select("source_id, content, metadata, created_at")
        .eq("notebook_id", notebook_id)
        .order("created_at")
        .limit(limit)
        .execute()
    )
    return result.data or []


def _sentences(text: str) -> list[str]:
    candidates = re.split(r"(?<=[.!?])\s+", text.strip())
    cleaned = [s.strip() for s in candidates if len(s.strip()) > 40]
    return cleaned


def _fallback_script(notebook_id: str, chunks: list[dict]) -> str:
    grouped: dict[str, list[str]] = {}
    for row in chunks:
        source = row.get("source_id") or "unknown"
        grouped.setdefault(source, []).append(row.get("content") or "")

    lines = [
        "# Podcast Script",
        "",
        "## Intro",
        "Host 1: Welcome back! Today we're diving into this notebook's key ideas.",
        "Host 2: We'll walk through the main sources and end with practical takeaways.",
        "",
    ]

    for index, (source, pieces) in enumerate(grouped.items(), start=1):
        merged = " ".join(pieces)
        summary_sentences = _sentences(merged)[:3]
        lines.append(f"## Segment {index}: {source}")
        if summary_sentences:
            for sent in summary_sentences:
                lines.append(f"- {sent}")
        else:
            lines.append("- This source was ingested but had limited extractable text.")
        lines.append("")

    lines.extend(
        [
            "## Key Takeaways",
            "- The notebook combines multiple sources into a coherent set of ideas.",
            "- Compare perspectives across sources before drawing conclusions.",
            "- Use this script as a draft and tailor it to your audience.",
            "",
            "## Outro",
            f"Host 1: That wraps up our notebook walkthrough for {notebook_id}.",
            "Host 2: Thanks for listening — see you in the next episode!",
        ]
    )

    return "\n".join(lines)


def _llm_script(notebook_id: str, chunks: list[dict]) -> str:
    if not HF_TOKEN:
        return _fallback_script(notebook_id, chunks)

    context_parts = []
    total_chars = 0
    char_cap = 18000
    for row in chunks:
        content = (row.get("content") or "").strip()
        source = row.get("source_id") or "unknown"
        snippet = f"[{source}] {content}"
        if total_chars + len(snippet) > char_cap:
            break
        context_parts.append(snippet)
        total_chars += len(snippet)

    context = "\n\n".join(context_parts)
    prompt = f"""
You are creating a podcast script from notebook materials.
Write a clear, engaging script with these sections:
1) Intro
2) 3-6 discussion segments
3) Key Takeaways
4) Outro

Rules:
- Keep it factual and grounded in provided content.
- Use two hosts: Host 1 and Host 2.
- Mention source names when relevant.
- Output markdown only.

Notebook ID: {notebook_id}

Context:
{context}
""".strip()

    try:
        client = InferenceClient(token=HF_TOKEN)
        response = client.chat_completion(
            model=HF_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1400,
            temperature=0.3,
        )
        text = (response.choices[0].message.content or "").strip()
        return text if text else _fallback_script(notebook_id, chunks)
    except Exception:
        return _fallback_script(notebook_id, chunks)


def generate_podcast(notebook_id: str, user_id: str) -> dict:
    """Generate podcast markdown from notebook chunks and persist artifact."""
    chunks = _get_notebook_chunks(notebook_id)
    if not chunks:
        raise ValueError("No notebook content found. Add text/PDF/URL sources first.")

    script = _llm_script(notebook_id, chunks)

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"podcast_{stamp}.md"
    storage_path = f"{get_artifacts_path(user_id, notebook_id)}/{filename}"

    save_file(storage_path, script)
    artifact = create_artifact(notebook_id=notebook_id, type="podcast", storage_path=storage_path)

    return {
        "artifact_id": artifact["id"] if artifact else None,
        "storage_path": storage_path,
        "script": script,
        "sources_count": len({row.get('source_id') for row in chunks}),
        "chunks_used": len(chunks),
    }


def _normalize_script_for_tts(script: str) -> str:
    cleaned = re.sub(r"[#*_`>-]", " ", script)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _tts_request(text: str) -> bytes:
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN is required for TTS generation.")

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "audio/wav",
    }
    payload_variants = [{"inputs": text}, {"text_inputs": text}]

    errors = []
    base_urls = [
        "https://router.huggingface.co/hf-inference/models",
        "https://api-inference.huggingface.co/models",
    ]

    for model_id in _tts_models():
        client = InferenceClient(token=HF_TOKEN)
        try:
            audio = client.text_to_speech(text, model=model_id)
            if isinstance(audio, (bytes, bytearray)) and audio:
                return bytes(audio)
            if hasattr(audio, "read"):
                data = audio.read()
                if data:
                    return data
            errors.append(f"{model_id}: client returned empty audio")
        except Exception as exc:
            errors.append(f"{model_id}: InferenceClient {exc}")

        for base_url in base_urls:
            url = f"{base_url}/{model_id}"
            for payload in payload_variants:
                try:
                    response = requests.post(url, headers=headers, json=payload, timeout=180)
                    if response.status_code in (404, 410):
                        errors.append(f"{model_id} @ {base_url}: HTTP {response.status_code}")
                        continue

                    response.raise_for_status()
                    if response.content:
                        return response.content

                    errors.append(f"{model_id} @ {base_url}: empty audio response")
                except requests.RequestException as exc:
                    errors.append(f"{model_id} @ {base_url}: {exc}")

    raise ValueError(
        "TTS generation failed for all candidate models. "
        f"Tried: {', '.join(_tts_models())}. "
        f"Details: {' | '.join(errors[:3])}"
    )


def _concat_wav_bytes(parts: list[bytes]) -> bytes:
    if not parts:
        raise ValueError("No audio segments returned from TTS model.")

    output_buffer = io.BytesIO()
    output_wave = None
    params = None

    try:
        for index, part in enumerate(parts):
            with wave.open(io.BytesIO(part), "rb") as segment:
                if index == 0:
                    params = segment.getparams()
                    output_wave = wave.open(output_buffer, "wb")
                    output_wave.setparams(params)
                target_channels = params.nchannels
                target_width = params.sampwidth
                target_rate = params.framerate

                input_frames = segment.readframes(segment.getnframes())
                input_width = segment.getsampwidth()
                input_channels = segment.getnchannels()
                input_rate = segment.getframerate()

                if input_width != target_width:
                    input_frames = audioop.lin2lin(input_frames, input_width, target_width)
                    input_width = target_width

                if input_channels != target_channels:
                    if input_channels == 2 and target_channels == 1:
                        input_frames = audioop.tomono(input_frames, input_width, 0.5, 0.5)
                    elif input_channels == 1 and target_channels == 2:
                        input_frames = audioop.tostereo(input_frames, input_width, 1.0, 1.0)
                    else:
                        raise ValueError(
                            f"Unsupported channel conversion from {input_channels} to {target_channels}"
                        )
                    input_channels = target_channels

                if input_rate != target_rate:
                    input_frames, _ = audioop.ratecv(
                        input_frames,
                        input_width,
                        input_channels,
                        input_rate,
                        target_rate,
                        None,
                    )

                output_wave.writeframes(input_frames)
    finally:
        if output_wave is not None:
            output_wave.close()

    return output_buffer.getvalue()


def _is_valid_wav(audio_bytes: bytes) -> bool:
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
            return wav_file.getnframes() > 0 and wav_file.getframerate() > 0
    except Exception:
        return False


def _aiff_to_wav_bytes(aiff_path: Path) -> bytes:
    with aifc.open(str(aiff_path), "rb") as aiff_file:
        nchannels = aiff_file.getnchannels()
        sampwidth = aiff_file.getsampwidth()
        framerate = aiff_file.getframerate()
        frames = aiff_file.readframes(aiff_file.getnframes())

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(nchannels)
        wav_file.setsampwidth(sampwidth)
        wav_file.setframerate(framerate)
        wav_file.writeframes(frames)
    return buffer.getvalue()


def _synthesize_local_tts_with_say(text: str) -> tuple[bytes, str]:
    say_bin = shutil.which("say")
    if not say_bin:
        raise ValueError("macOS 'say' command not found")
    afconvert_bin = shutil.which("afconvert")

    parts: list[bytes] = []
    chunks = textwrap.wrap(text, width=700, break_long_words=False, break_on_hyphens=False)
    if not chunks:
        raise ValueError("No text provided for macOS say generation")

    with tempfile.TemporaryDirectory() as temp_dir:
        for index, chunk in enumerate(chunks):
            aiff_path = Path(temp_dir) / f"tts_{index}.aiff"
            subprocess.run(
                [say_bin, "-o", str(aiff_path), chunk],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )

            if not aiff_path.exists() or aiff_path.stat().st_size == 0:
                raise ValueError("macOS say produced empty audio")

            wav_bytes = None
            if afconvert_bin:
                wav_path = Path(temp_dir) / f"tts_{index}.wav"
                subprocess.run(
                    [afconvert_bin, "-f", "WAVE", "-d", "LEI16", "-r", "22050", str(aiff_path), str(wav_path)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if wav_path.exists() and wav_path.stat().st_size > 0:
                    wav_bytes = wav_path.read_bytes()

            if not wav_bytes:
                wav_bytes = _aiff_to_wav_bytes(aiff_path)

            if wav_bytes and _is_valid_wav(wav_bytes):
                parts.append(wav_bytes)
                continue

            raise ValueError("macOS say WAV conversion failed")

    combined = _concat_wav_bytes(parts)
    if not _is_valid_wav(combined):
        raise ValueError("macOS say fallback produced invalid WAV")
    return combined, ".wav"


def _synthesize_tts_audio(script: str) -> tuple[bytes, str]:
    normalized = _normalize_script_for_tts(script)
    if not normalized:
        raise ValueError("Podcast script is empty.")

    chunks = textwrap.wrap(normalized, width=800, break_long_words=False, break_on_hyphens=False)
    try:
        audio_parts = [_tts_request(chunk) for chunk in chunks]
        cloud_audio = _concat_wav_bytes(audio_parts)
        if _is_valid_wav(cloud_audio):
            return cloud_audio, ".wav"
        raise ValueError("Cloud TTS returned invalid/empty WAV")
    except Exception as cloud_error:
        transformers_error = None
        try:
            transformers_audio = _synthesize_local_tts_with_transformers(normalized)
            if _is_valid_wav(transformers_audio):
                return transformers_audio, ".wav"
            raise ValueError("SpeechT5 produced invalid WAV")
        except Exception as exc:
            transformers_error = exc

        say_error = None
        try:
            say_audio, say_ext = _synthesize_local_tts_with_say(normalized)
            if say_audio:
                return say_audio, say_ext
            raise ValueError("macOS say produced empty audio")
        except Exception as exc:
            say_error = exc

        try:
            engine = pyttsx3.init()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_path = temp_audio.name

            try:
                engine.save_to_file(normalized, temp_path)
                engine.runAndWait()
                data = Path(temp_path).read_bytes()
                if not _is_valid_wav(data):
                    raise ValueError("Local TTS produced invalid/empty WAV.")
                return data, ".wav"
            finally:
                temp_file = Path(temp_path)
                if temp_file.exists():
                    temp_file.unlink()
        except Exception as local_error:
            raise ValueError(
                f"Cloud TTS failed ({cloud_error}); transformers fallback failed ({transformers_error}); "
                f"say fallback failed ({say_error}); local fallback failed ({local_error})."
            )


def generate_podcast_audio(notebook_id: str, user_id: str, script: str) -> dict:
    """Generate WAV audio from a podcast script and persist as artifact."""
    audio_bytes, extension = _synthesize_tts_audio(script)

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"podcast_{stamp}{extension}"
    storage_path = f"{get_artifacts_path(user_id, notebook_id)}/{filename}"

    save_file(storage_path, audio_bytes)
    artifact = create_artifact(notebook_id=notebook_id, type="podcast_audio", storage_path=storage_path)

    local_cache_dir = Path("data") / "artifacts" / user_id / str(notebook_id)
    local_cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_cache_dir / filename
    local_path.write_bytes(audio_bytes)

    return {
        "artifact_id": artifact["id"] if artifact else None,
        "storage_path": storage_path,
        "audio_path": str(local_path),
    }
