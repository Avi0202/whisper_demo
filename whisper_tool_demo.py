import os
import sys
import traceback
import tempfile
import logging
import asyncio
from logging.handlers import RotatingFileHandler
import yt_dlp
import ffmpeg
from openai import AsyncOpenAI
from fastapi import HTTPException
import streamlit as st
from dotenv import load_dotenv
import re
import glob

# Load environment variables
load_dotenv()

# =========================================================
# LOGGER
# =========================================================
async def get_logger(name: str) -> logging.Logger:
    LOG_DIR = "logs"
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_FILE = os.path.join(LOG_DIR, "whisper_diarized.log")
    LOG_FORMAT = "%(levelname)s | %(asctime)s | %(name)s | %(message)s"

    logger = logging.getLogger(name)
    if not logger.handlers:
        fh = RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(logging.Formatter(LOG_FORMAT))
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter(LOG_FORMAT))
        for h in (fh, ch):
            h.setLevel(logging.INFO)
            logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


# =========================================================
# PHONETIC ENFORCER
# =========================================================
async def ensure_english_phonetic(text: str, api_key: str) -> str:
    """
    Ensures all text is in English letters only.
    If not purely alphabetic (A–Z), sends it to gpt-4o-mini for phonetic rewriting.
    """
    if not text or text.strip() == "":
        return text

    # If text already looks English, keep it
    if re.fullmatch(r"[A-Za-z0-9\s.,!?'\-]+", text):
        return text

    try:
        client = AsyncOpenAI(api_key=api_key)
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a phonetic transliteration assistant. "
                        "Rewrite the following text using only English letters to represent (No latin characters allowed) "
                        "the pronunciation. Do not translate it to English meaning, only render its sounds in English script without using any non ASCII characters."
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0.2,
            max_tokens=2000,
        )
        result = response.choices[0].message.content.strip()
        return result
    except Exception as e:
        print(f"[romanize-fallback] Failed: {e}")
        return text


# =========================================================
# DOWNLOAD + COMPRESS
# =========================================================
async def download_audio_from_youtube(url: str) -> str:
    import subprocess
    import shlex
    import shutil
    import youtube_dl
    from pytube import YouTube

    tmpdir = tempfile.mkdtemp(prefix="yt_audio_")
    output_template = os.path.join(tmpdir, "audio")

    def _run():
        # 1️⃣ yt-dlp
        try:
            ydl_opts = {
                "format": "bestaudio[ext=m4a]/bestaudio/best",
                "outtmpl": output_template,
                "postprocessors": [
                    {"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "192"}
                ],
                "noplaylist": True,
                "quiet": True,
                "source_address": "0.0.0.0",
                "retries": 1,
                "nocheckcertificate": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            for f in os.listdir(tmpdir):
                if f.endswith(".wav"):
                    return os.path.join(tmpdir, f)
            raise FileNotFoundError("yt-dlp finished but no .wav found")
        except Exception as e:
            print(f"[yt-dlp] ❌ Failed: {e}")

        # 2️⃣ youtube_dl
        try:
            print("[youtube_dl] Trying fallback...")
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": output_template,
                "postprocessors": [
                    {"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "192"}
                ],
                "noplaylist": True,
                "quiet": True,
            }
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            for f in os.listdir(tmpdir):
                if f.endswith(".wav"):
                    return os.path.join(tmpdir, f)
            raise FileNotFoundError("youtube_dl finished but no .wav found")
        except Exception as e:
            print(f"[youtube_dl] ❌ Failed: {e}")

        # 3️⃣ pytube
        try:
            print("[pytube] Trying fallback...")
            yt = YouTube(url)
            stream = yt.streams.filter(only_audio=True).first()
            if not stream:
                raise ValueError("No audio streams found via pytube.")
            out_file = stream.download(output_path=tmpdir, filename="audio.mp4")
            wav_path = os.path.join(tmpdir, "audio.wav")
            (ffmpeg.input(out_file).output(wav_path, ac=1, ar=16000).overwrite_output().run(quiet=True))
            if os.path.exists(wav_path):
                return wav_path
            raise FileNotFoundError("pytube finished but no .wav found")
        except Exception as e:
            print(f"[pytube] ❌ Failed: {e}")

        # 4️⃣ ffmpeg stream grab
        try:
            print("[ffmpeg] Trying direct stream grab...")
            ffmpeg_bin = shutil.which("ffmpeg")
            if not ffmpeg_bin:
                raise RuntimeError("ffmpeg not installed or not in PATH")
            fallback_out = os.path.join(tmpdir, "fallback_audio.wav")
            cmd = f'{ffmpeg_bin} -y -i "{url}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{fallback_out}"'
            result = subprocess.run(shlex.split(cmd), capture_output=True, text=True, timeout=60)
            if result.returncode == 0 and os.path.exists(fallback_out):
                return fallback_out
            raise RuntimeError(result.stderr.strip() or "ffmpeg could not extract audio stream")
        except Exception as e:
            print(f"[ffmpeg] ❌ Failed: {e}")

        raise RuntimeError("All extraction methods failed: yt-dlp, youtube_dl, pytube, and ffmpeg")

    return await asyncio.to_thread(_run)


async def compress_audio(path: str, br: str = "64k", sr: int = 16000) -> str:
    base, _ = os.path.splitext(path)
    out = base + "_compressed.wav"

    def _run():
        (ffmpeg.input(path).output(out, **{"b:a": br, "ac": 1, "ar": sr, "y": None}).overwrite_output().run(quiet=True))
        return out

    return await asyncio.to_thread(_run)


# =========================================================
# MAIN FUNCTION
# =========================================================

import tempfile


async def whisper_diarized(youtube_url: str, api_key: str, flag: bool = False):
    logger = await get_logger("whisper_diarized")
    logger.info("=== Start diarized transcription ===")

    try:
        # 1️⃣ Download audio
        audio_path = await download_audio_from_youtube(youtube_url)
        logger.info(f"Downloaded {audio_path}")

        # 2️⃣ Always compress once
        logger.info("Compressing audio…")
        audio_path = await compress_audio(audio_path)
        size = os.path.getsize(audio_path) / 1e6
        logger.info(f"Compressed size: {size:.2f} MB")

        client = AsyncOpenAI(api_key=api_key)
        model_name = "gpt-4o-transcribe-diarize" if flag else "gpt-4o-transcribe"

        # 3️⃣ If audio still large, split with ffmpeg
        if size > 25:
            logger.info("⚠️ File too large, splitting into chunks for streaming (using ffmpeg segment).")
            chunk_dir = tempfile.mkdtemp(prefix="chunks_")
            seg_pattern = os.path.join(chunk_dir, "part_%03d.wav")
            segment_seconds = 180  # 3 min chunks (adjust if needed)

            def _split_with_ffmpeg():
                (
                    ffmpeg
                    .input(audio_path)
                    .output(
                        seg_pattern,
                        f="segment",
                        segment_time=segment_seconds,
                        reset_timestamps=1,
                        ac=1,
                        ar=16000,
                        **{"map": "0"}
                    )
                    .overwrite_output()
                    .run(quiet=True)
                )
                return sorted(glob.glob(os.path.join(chunk_dir, "part_*.wav")))

            chunks = await asyncio.to_thread(_split_with_ffmpeg)
            if not chunks:
                raise RuntimeError("FFmpeg produced no chunks")

            logger.info(f"Split into {len(chunks)} chunks via ffmpeg.")

            transcripts = []

            # 4️⃣ Process chunks sequentially
            for i, path in enumerate(chunks):
                logger.info(f"Transcribing chunk {i+1}/{len(chunks)} …")
                with open(path, "rb") as f:
                    response = await client.audio.transcriptions.create(
                        model=model_name,
                        file=f,
                        response_format="diarized_json" if flag else "json",
                    )

                if flag:
                    for s in response.segments:
                        raw_text = getattr(s, "text", "") or ""
                        s.text = await ensure_english_phonetic(raw_text, api_key)
                    transcripts.extend(response.segments)
                else:
                    raw_text = getattr(response, "text", "") or ""
                    text = await ensure_english_phonetic(raw_text, api_key)
                    transcripts.append(dict(text=text))

            # 5️⃣ Merge final output
            if flag:
                summary = " ".join(
                    f"{getattr(s, 'speaker', None)}: {s.text}"
                    for s in transcripts if getattr(s, "text", None)
                )
                result_segments = transcripts
            else:
                summary = " ".join(t["text"] for t in transcripts)
                result_segments = transcripts

        # 6️⃣ If small enough, process normally
        else:
            logger.info("Processing normally …")
            with open(audio_path, "rb") as f:
                response = await client.audio.transcriptions.create(
                    model=model_name,
                    file=f,
                    response_format="diarized_json" if flag else "json",
                    chunking_strategy="auto",
                )

            if flag:
                result_segments = []
                for s in response.segments:
                    raw_text = getattr(s, "text", "").strip()
                    phonetic_text = await ensure_english_phonetic(raw_text, api_key)
                    result_segments.append({
                        "speaker": getattr(s, "speaker", None),
                        "start": getattr(s, "start", None),
                        "end": getattr(s, "end", None),
                        "text": phonetic_text,
                    })
                summary = " ".join(f"{s['speaker']}: {s['text']}" for s in result_segments)
            else:
                raw_text = getattr(response, "text", "").strip()
                text = await ensure_english_phonetic(raw_text, api_key)
                result_segments = [dict(text=text)]
                summary = text

        # 7️⃣ Done
        logger.info("=== Completed transcription ===")
        return {"model": model_name, "segments": result_segments, "summary": summary}

    except Exception as e:
        logger.error(f"FAILED: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================
# STREAMLIT UI
# =========================================================
st.set_page_config(page_title="🎙️ YouTube Whisper Diarizer", layout="centered")

st.title("🎧 YouTube → Whisper Transcriber")
st.caption("Transcribe any YouTube video using OpenAI’s Whisper. Automatically converts non-English to phonetic English (no translation).")

youtube_url = st.text_input("🔗 Enter YouTube video URL", placeholder="https://www.youtube.com/watch?v=...")
flag = st.checkbox("🗣️ Enable Speaker Diarization")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.warning("⚠️ OPENAI_API_KEY not found in environment.")
else:
    st.success("✅ Using API key from environment.")

if st.button("🚀 Transcribe"):
    if not youtube_url:
        st.error("Please enter a valid YouTube link.")
        st.stop()
    with st.spinner("Processing... please wait ⏳"):
        try:
            result = asyncio.run(whisper_diarized(youtube_url, api_key, flag=flag))
            st.success("✅ Done!")

            st.subheader("🧠 Model Used:")
            st.code(result["model"], language="text")

            st.subheader("📜 English-Only Transcript:")
            st.text_area("Transcript", value=result["summary"], height=300)

            if flag:
                with st.expander("🧩 Detailed Segments"):
                    for seg in result["segments"]:
                        st.markdown(f"**{seg['speaker']}** [{seg['start']}–{seg['end']}s]: {seg['text']}")
        except Exception as e:
            st.error(f"❌ Error: {e}")