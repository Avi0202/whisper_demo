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
from dotenv import load_dotenv  # 👈 add this

# Load environment variables from .env file
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
# DOWNLOAD + COMPRESS
# =========================================================
async def download_audio_from_youtube(url: str) -> str:
    tmpdir = tempfile.mkdtemp(prefix="yt_audio_")
    output_template = os.path.join(tmpdir, "audio")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [{"key": "FFmpegExtractAudio",
                            "preferredcodec": "wav",
                            "preferredquality": "192"}],
        "quiet": True,
    }

    def _run():
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        for f in os.listdir(tmpdir):
            if f.endswith(".wav"):
                return os.path.join(tmpdir, f)
        raise FileNotFoundError("wav not found")
    return await asyncio.to_thread(_run)


async def compress_audio(path: str, br: str = "64k", sr: int = 16000) -> str:
    base, _ = os.path.splitext(path)
    out = base + "_compressed.wav"

    def _run():
        (
            ffmpeg.input(path)
            .output(out, **{"b:a": br, "ac": 1, "ar": sr, "y": None})
            .overwrite_output()
            .run(quiet=True)
        )
        return out
    return await asyncio.to_thread(_run)


# =========================================================
# MAIN FUNCTION
# =========================================================
async def whisper_diarized(youtube_url: str, api_key: str, flag: bool=False):
    logger = await get_logger("whisper_diarized")
    MAX_MB = 25
    logger.info("=== Start diarized transcription ===")

    try:
        # 1. Download
        audio_path = await download_audio_from_youtube(youtube_url)
        size = os.path.getsize(audio_path) / 1e6
        logger.info(f"Downloaded {audio_path} ({size:.2f} MB)")
    
        # 2. Compress if needed
        if size > MAX_MB:
            logger.info("Compressing …")
            audio_path = await compress_audio(audio_path)
            size = os.path.getsize(audio_path) / 1e6
            logger.info(f"Compressed size {size:.2f} MB")
        else:
            logger.info("Compression not needed")
    
        # 3. Transcription
        logger.info("Calling OpenAI transcription API …")
        client = AsyncOpenAI(api_key=api_key)
        model_name = "gpt-4o-transcribe-diarize" if flag else "gpt-4o-transcribe"
    
        with open(audio_path, "rb") as f:
            response = await client.audio.transcriptions.create(
                model=model_name,
                file=f,
                response_format="diarized_json" if flag else "json",
                chunking_strategy="auto",
            )
    
        segments = []
        summary = ""
    
        if flag:
            if not hasattr(response, "segments") or not response.segments:
                raise HTTPException(500, "No segments in diarized response")
    
            segments = [
                {
                    "speaker": getattr(s, "speaker", None),
                    "start": getattr(s, "start", None),
                    "end": getattr(s, "end", None),
                    "text": getattr(s, "text", "").strip(),
                }
                for s in response.segments
            ]
    
            summary = " ".join(f"{s['speaker']}: {s['text']}" for s in segments if s["text"])
    
        else:
            text = getattr(response, "text", "").strip()
            if not text:
                raise HTTPException(500, "Empty transcription text in response")
    
            segments = [dict(speaker=None, start=None, end=None, text=text)]
            summary = text
    
        logger.info(f"=== Completed {'diarized' if flag else 'standard'} transcription ===")
    
        return {"model": model_name, "segments": segments, "summary": summary}
    
    except Exception as e:
        logger.error(f"FAILED: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================
# STREAMLIT UI
# =========================================================
st.set_page_config(page_title="🎙️ YouTube Whisper Diarizer", layout="centered")

st.title("🎧 YouTube → Whisper Transcriber")
st.caption("Transcribe any YouTube video using OpenAI’s Whisper (with optional diarization).")

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

            st.subheader("📜 Transcript:")
            st.text_area("Transcript", value=result["summary"], height=300)

            if flag:
                with st.expander("🧩 Detailed Segments"):
                    for seg in result["segments"]:
                        st.markdown(f"**{seg['speaker']}** [{seg['start']}–{seg['end']}s]: {seg['text']}")
        except Exception as e:
            st.error(f"❌ Error: {e}")