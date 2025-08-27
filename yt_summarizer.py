#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YouTube Transcript & Summary Agent (CLI)
---------------------------------------

Features
- Extract transcript using youtube-transcript-api (no API keys required).
- (Optional) If no transcript exists, try to download audio with pytube and
  transcribe using open-source Whisper (local) — requires ffmpeg installed.
- Summarize long transcripts robustly using Hugging Face transformers
  (default: "facebook/bart-large-cnn").
- Language-aware summaries (e.g., Persian "fa" or English "en").
- Map–reduce chunking to avoid token limits.

Usage
  python yt_summarizer.py --url https://www.youtube.com/watch?v=VIDEO_ID \
      --lang fa --max-chars 900 --model facebook/bart-large-cnn \
      --out summary.txt

  # If you want OpenAI summarization instead of transformers, set --use-openai
  # and provide OPENAI_API_KEY in env. (Optional path.)

Install
  pip install youtube-transcript-api pytube transformers torch tqdm
  # Optional, only if you want fallback transcription:
  pip install openai-whisper
  # And ensure ffmpeg is installed on your system.

Note
- This script does NOT require any Google/YouTube API keys for transcripts.
- Some videos have disabled transcripts; in that case, use the Whisper fallback.
"""
from __future__ import annotations
import argparse
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from typing import List, Optional

# Third-party
from tqdm import tqdm

# Lazy imports to speed up startup

def lazy_import_transformers():
    import importlib
    return importlib.import_module("transformers")

def lazy_import_youtube_transcript_api():
    import importlib
    return importlib.import_module("youtube_transcript_api")

def lazy_import_pytube():
    import importlib
    return importlib.import_module("pytube")

def lazy_import_whisper():
    import importlib
    return importlib.import_module("whisper")

# ----------------------------- Utilities ---------------------------------

YOUTUBE_ID_PATTERN = re.compile(r"(?:v=|be/|embed/)([A-Za-z0-9_-]{11})")


def extract_video_id(url_or_id: str) -> str:
    """Extract a YouTube video ID from a URL or return the input if it looks like an ID."""
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url_or_id):
        return url_or_id
    m = YOUTUBE_ID_PATTERN.search(url_or_id)
    if not m:
        raise ValueError("Could not detect YouTube video ID. Provide a valid URL or 11-char ID.")
    return m.group(1)


@dataclass
class TranscriptResult:
    source: str  # "captions" or "whisper"
    text: str


# ----------------------------- Transcript ---------------------------------

def fetch_transcript(video_id: str, languages: Optional[List[str]] = None) -> Optional[TranscriptResult]:
    """Try to fetch transcript via youtube-transcript-api. Returns None if unavailable."""
    yta = lazy_import_youtube_transcript_api()
    if languages is None:
        # Try requested lang first, then English, auto, etc.
        languages = ["en", "en-US", "fa", "auto"]
    try:
        transcripts = yta.YouTubeTranscriptApi.list_transcripts(video_id)
        # Prefer manually created transcripts in the given languages order
        for lang in languages:
            try:
                t = transcripts.find_manually_created_transcript([lang])
                items = t.fetch()
                text = " ".join([seg["text"].strip() for seg in items if seg.get("text")])
                if text:
                    return TranscriptResult(source="captions", text=text)
            except Exception:
                pass
        # Fallback to auto-generated if allowed
        for lang in languages:
            try:
                t = transcripts.find_generated_transcript([lang])
                items = t.fetch()
                text = " ".join([seg["text"].strip() for seg in items if seg.get("text")])
                if text:
                    return TranscriptResult(source="captions", text=text)
            except Exception:
                pass
    except Exception:
        return None
    return None


def whisper_fallback(video_id: str, model_name: str = "small") -> Optional[TranscriptResult]:
    """Download audio via pytube, transcribe locally with Whisper. Requires ffmpeg.
       model_name: tiny/base/small/medium/large. 'small' is a good tradeoff.
    """
    try:
        pytube = lazy_import_pytube()
        whisper = lazy_import_whisper()
    except Exception as e:
        sys.stderr.write(f"Whisper fallback unavailable: {e}\n")
        return None

    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        yt = pytube.YouTube(url)
        # Get audio-only stream
        stream = yt.streams.filter(only_audio=True).first()
        if not stream:
            return None
        with tempfile.TemporaryDirectory() as td:
            out_file = stream.download(output_path=td, filename_prefix="audio_")
            model = whisper.load_model(model_name)
            result = model.transcribe(out_file)
            text = result.get("text", "").strip()
            if text:
                return TranscriptResult(source="whisper", text=text)
    except Exception as e:
        sys.stderr.write(f"Whisper fallback failed: {e}\n")
        return None
    return None

# ----------------------------- Summarization -------------------------------

def split_into_chunks(text: str, max_chars: int = 4000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks to preserve context across boundaries."""
    text = re.sub(r"\s+", " ", text).strip()
    chunks: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + max_chars, n)
        # Try to break at a sentence boundary
        slice_ = text[i:end]
        last_period = slice_.rfind(".")
        if last_period > 0 and end != n:
            end = i + last_period + 1
            slice_ = text[i:end]
        chunks.append(slice_)
        i = max(end - overlap, i + 1)
    return chunks


def summarize_with_transformers(chunks: List[str], model_name: str = "facebook/bart-large-cnn",
                                lang: str = "en", max_chars: int = 900) -> str:
    """Map–reduce summarization using HF transformers.
       lang controls the *prompting* for non-English summaries.
    """
    transformers = lazy_import_transformers()
    from transformers import pipeline

    # Create a summarization pipeline
    summarizer = pipeline("summarization", model=model_name)

    partial_summaries: List[str] = []
    for chunk in tqdm(chunks, desc="Summarizing chunks"):
        # For multilingual targets, lightly prompt the model via prefix.
        if lang != "en":
            prefixed = f"Summarize the following content in {lang}:\n{chunk}"
        else:
            prefixed = chunk
        # HF pipelines handle internal tokenization; set sensible lengths
        out = summarizer(prefixed, max_length=300, min_length=120, do_sample=False)
        partial_summaries.append(out[0]["summary_text"]) 

    # Reduce step
    joined = "\n".join(partial_summaries)
    if lang != "en":
        reduce_prompt = (
            f"You are a helpful assistant. Create a cohesive, bullet-rich summary in {lang} "
            f"from the notes below. Highlight key points, claims, numbers, and any step-by-step lists.\n\n{joined}"
        )
    else:
        reduce_prompt = (
            "Create a cohesive, bullet-rich executive summary from the notes below. "
            "Highlight key points, claims, numbers, and any step-by-step lists.\n\n" + joined
        )
    final = summarizer(reduce_prompt, max_length=300, min_length=120, do_sample=False)
    final_text = final[0]["summary_text"].strip()

    # If still too long, clip to max_chars conservatively
    if len(final_text) > max_chars:
        final_text = final_text[:max_chars].rsplit(" ", 1)[0] + "…"
    return final_text


# Optional: OpenAI summarization path (requires env OPENAI_API_KEY)

def summarize_with_openai(chunks: List[str], lang: str = "en", max_chars: int = 900) -> str:
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai package not installed. `pip install openai`. ") from e
    client = OpenAI()

    partials = []
    sys_prompt = (
        "You are a world-class summarizer. Write concise, faithful summaries. "
        "Preserve key facts, numbers, names."
    )
    for ch in chunks:
        user_prompt = (
            (f"Summarize in {lang}. " if lang != "en" else "Summarize. ") +
            "Keep it factual and structured with bullets if appropriate.\n\n" + ch
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",  # any capable model
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        partials.append(resp.choices[0].message.content)

    reduce_prompt = (
        (f"In {lang}, " if lang != "en" else "") +
        "merge these notes into a single, cohesive summary with bullets for sections.\n\n" + "\n\n".join(partials)
    )
    final = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": reduce_prompt},
        ],
        temperature=0.2,
    )
    text = final.choices[0].message.content.strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "…"
    return text

# ----------------------------- Orchestrator --------------------------------

def run(url_or_id: str, lang: str = "fa", max_chars: int = 900, use_openai: bool = False,
        hf_model: str = "facebook/bart-large-cnn", whisper_model: str = "small") -> str:
    vid = extract_video_id(url_or_id)

    # 1) Try captions first
    tr = fetch_transcript(vid, languages=[lang, "en", "fa", "auto"])

    # 2) Fallback to Whisper
    if tr is None:
        tr = whisper_fallback(vid, model_name=whisper_model)
        if tr is None:
            raise RuntimeError("No transcript available and Whisper fallback failed.")

    # 3) Chunk
    chunks = split_into_chunks(tr.text, max_chars=3500, overlap=250)

    # 4) Summarize
    if use_openai:
        summary = summarize_with_openai(chunks, lang=lang, max_chars=max_chars)
    else:
        summary = summarize_with_transformers(chunks, model_name=hf_model, lang=lang, max_chars=max_chars)

    header = (
        f"Source: {tr.source}\n" 
        f"Video: https://www.youtube.com/watch?v={vid}\n"
        f"Lang: {lang}\n\n"
    )
    return header + summary


def main():
    parser = argparse.ArgumentParser(description="YouTube Transcript & Summary Agent")
    parser.add_argument("--url", required=True, help="YouTube URL or 11-char video ID")
    parser.add_argument("--lang", default="fa", help="Output summary language, e.g., fa or en")
    parser.add_argument("--max-chars", type=int, default=900, help="Max characters for the final summary")
    parser.add_argument("--model", dest="hf_model", default="facebook/bart-large-cnn", help="HF summarization model")
    parser.add_argument("--whisper", dest="whisper_model", default="small", help="Whisper model size for fallback")
    parser.add_argument("--use-openai", action="store_true", help="Use OpenAI for summarization instead of transformers")
    parser.add_argument("--out", default="", help="Optional path to save the summary")
    args = parser.parse_args()

    try:
        text = run(
            url_or_id=args.url,
            lang=args.lang,
            max_chars=args.max_chars,
            use_openai=args.use_openai,
            hf_model=args.hf_model,
            whisper_model=args.whisper_model,
        )
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Saved -> {args.out}")
        else:
            print(text)
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
