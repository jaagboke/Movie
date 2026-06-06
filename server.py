"""
Audiobook Web App — Backend Server
===================================
Run:  python server.py
Then open: http://localhost:5000

Install:
    pip install flask edge-tts pdfplumber ebooklib beautifulsoup4 pydub
"""

import asyncio
import glob
import json
import os
import re
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory

app = Flask(__name__, static_folder="static")

UPLOAD_DIR  = "uploads"
AUDIO_DIR   = "audio"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR,  exist_ok=True)

# ── In-memory job tracker ─────────────────────────────────────────────────────
jobs = {}   # job_id → { status, progress, total, chapters, error }


# ── Text extraction ───────────────────────────────────────────────────────────

def extract_from_pdf(filepath):
    import pdfplumber
    pages = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n".join(pages)


def extract_from_epub(filepath):
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
    book = epub.read_epub(filepath)
    parts = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            text = soup.get_text(separator="\n")
            if text.strip():
                parts.append(text)
    return "\n\n".join(parts)


def extract_from_txt(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_text(filepath):
    ext = Path(filepath).suffix.lower()
    if ext == ".pdf":
        return extract_from_pdf(filepath)
    elif ext == ".epub":
        return extract_from_epub(filepath)
    elif ext == ".txt":
        return extract_from_txt(filepath)
    raise ValueError(f"Unsupported format: {ext}")


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text):
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*(www\.|http)\S+\s*$', '', text, flags=re.MULTILINE)
    return text.strip()


# ── Chapter splitting ─────────────────────────────────────────────────────────
#
# DAMA-DMBOK actual format (confirmed from screenshot):
#
#   CHAPTER 1          ← spaced caps, own line, no title on same line
#                      ← blank line / decorative rule
#   Data Management    ← title appears 1-3 lines later
#
# The regex must:
#   1. Match "CHAPTER <N>" on its own line
#   2. Look ahead up to 3 lines to grab the title
#   3. Ignore TOC hits (deduplicate by keeping last occurrence of each number)

# Step 1: find every "CHAPTER N" line (just the number, no title on same line)
CHAPTER_NUM_RE = re.compile(
    r'(?:^|\n)[ \t]*C\s*H\s*A\s*P\s*T\s*E\s*R[ \t]+(\d+)[ \t]*(?:\r?\n)',
    re.IGNORECASE
)

# Step 2: also handle inline "Chapter 1: Title" as fallback for other books
CHAPTER_INLINE_RE = re.compile(
    r'(?:^|\n)[ \t]*Chapter[ \t]+(\d+)[ \t]*[:–-][ \t]*([^\n]{3,80})',
    re.IGNORECASE
)


def slugify(text):
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'\s+', '_', text.strip())
    return text[:50]


def _grab_title_after(full_text, pos_after_num_line):
    """
    Look at the next few lines after "CHAPTER N" to find the title.
    Skips blank lines and lines that look like decorative rules (underscores, dashes).
    Returns the first real text line found, or empty string.
    """
    segment = full_text[pos_after_num_line: pos_after_num_line + 300]
    lines = segment.split('\n')
    for line in lines[:6]:
        stripped = line.strip()
        if not stripped:
            continue
        # Skip decorative lines: only underscores, dashes, equals
        if re.match(r'^[_\-=\s]+$', stripped):
            continue
        # Skip if it looks like a section number (e.g. "1. Introduction")
        if re.match(r'^\d+\.', stripped):
            continue
        # Skip very short lines (page numbers, single chars)
        if len(stripped) < 3:
            continue
        return stripped
    return ""


def find_chapter_positions(full_text):
    """
    Returns list of (start_pos, chapter_num, title) sorted by position.
    Deduplicates TOC hits by keeping only the LAST occurrence of each chapter number
    (TOC entries cluster at the top; real chapters are spread through the book).
    """
    candidates = []

    # Primary: spaced "CHAPTER N" on own line (DAMA-DMBOK style)
    for m in CHAPTER_NUM_RE.finditer(full_text):
        ch_num = int(m.group(1))
        title  = _grab_title_after(full_text, m.end())
        title  = re.sub(r'[\s_\.]+\d*\s*$', '', title).strip()
        if not title:
            title = f"Chapter {ch_num}"
        candidates.append((m.start(), ch_num, title, 'twoline'))

    # Fallback: "Chapter N: Title" inline (other book formats)
    if not candidates:
        for m in CHAPTER_INLINE_RE.finditer(full_text):
            ch_num = int(m.group(1))
            title  = m.group(2).strip()
            title  = re.sub(r'[\s_\.]+\d*\s*$', '', title).strip()
            candidates.append((m.start(), ch_num, title, 'inline'))

    if not candidates:
        return []

    candidates.sort(key=lambda x: x[0])

    # Keep only LAST occurrence of each chapter number → skips TOC
    seen = {}
    for pos, num, title, fmt in candidates:
        seen[num] = (pos, num, title)

    real = sorted(seen.values(), key=lambda x: x[0])
    return real


def split_into_chapters(full_text):
    positions = find_chapter_positions(full_text)

    if not positions:
        return []

    chapters = []
    for idx, (pos, ch_num, title) in enumerate(positions):
        next_pos = positions[idx + 1][0] if idx + 1 < len(positions) else len(full_text)

        # Skip past the heading block (CHAPTER N line + title line) to get body
        # Find end of the title line
        title_pos = full_text.find(title, pos)
        if title_pos != -1 and title_pos < next_pos:
            body_start = full_text.find('\n', title_pos) + 1
        else:
            body_start = full_text.find('\n', pos) + 1

        body = clean_text(full_text[body_start:next_pos])

        if len(body.split()) < 50:
            continue

        chapters.append({
            "number": ch_num,
            "title":  title,
            "text":   f"Chapter {ch_num}. {title}.\n\n{body}",
            "words":  len(body.split()),
        })

    return chapters


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text, chunk_size=4500):
    """
    Split text at sentence boundaries. Larger chunks = fewer API calls,
    better for heavy books like DAMA-DMBOK.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current, current_len = [], [], 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        # Hard cap: very long sentence -> split by words
        if len(sentence) > chunk_size:
            words = sentence.split()
            for word in words:
                if current_len + len(word) + 1 > chunk_size and current:
                    chunks.append(" ".join(current))
                    current, current_len = [word], len(word)
                else:
                    current.append(word)
                    current_len += len(word) + 1
            continue
        if current_len + len(sentence) > chunk_size and current:
            chunks.append(" ".join(current))
            current, current_len = [sentence], len(sentence)
        else:
            current.append(sentence)
            current_len += len(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks


# ── TTS ───────────────────────────────────────────────────────────────────────

async def tts_chunk(text, out_path, voice, rate, retries=3):
    import edge_tts
    for attempt in range(retries):
        try:
            tts = edge_tts.Communicate(text, voice=voice, rate=rate)
            await tts.save(out_path)
            return
        except Exception as e:
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                raise


async def tts_chapter(text, out_path, voice, rate, concurrency=3, progress_cb=None):
    """
    Convert one chapter to MP3. progress_cb(done, total) called after each chunk
    so the UI can show fine-grained progress inside heavy chapters.
    """
    chunks   = chunk_text(text)
    total    = len(chunks)
    temp_dir = out_path + "_tmp"
    os.makedirs(temp_dir, exist_ok=True)

    semaphore  = asyncio.Semaphore(concurrency)
    completed  = [0]

    async def run(i, chunk):
        async with semaphore:
            await tts_chunk(chunk, os.path.join(temp_dir, f"chunk_{i:05d}.mp3"), voice, rate)
            completed[0] += 1
            if progress_cb:
                progress_cb(completed[0], total)

    await asyncio.gather(*[run(i, c) for i, c in enumerate(chunks)])

    chunk_files = sorted(glob.glob(os.path.join(temp_dir, "chunk_*.mp3")))
    stitch(chunk_files, out_path)
    shutil.rmtree(temp_dir)


def stitch(chunk_files, out_path):
    """
    Concatenate MP3 chunks by appending raw bytes.
    No ffmpeg or pydub required — works everywhere.
    MP3 is a frame-based format so simple byte concatenation
    produces a valid, playable file.
    """
    with open(out_path, "wb") as out:
        for f in chunk_files:
            with open(f, "rb") as chunk:
                out.write(chunk.read())


# ── Background job ────────────────────────────────────────────────────────────

def run_job(job_id, filepath, voice, rate, book_audio_dir):
    try:
        jobs[job_id]["status"] = "extracting"

        full_text = extract_text(filepath)
        chapters  = split_into_chapters(full_text)

        if not chapters:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"]  = "No chapters detected. Check the PDF has 'CHAPTER N' headings."
            return

        jobs[job_id]["total"]    = len(chapters)
        jobs[job_id]["status"]   = "converting"
        jobs[job_id]["chapters"] = [
            {"number": ch["number"], "title": ch["title"], "words": ch["words"], "done": False}
            for ch in chapters
        ]

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        for idx, ch in enumerate(chapters):
            slug     = slugify(ch["title"])
            out_path = os.path.join(book_audio_dir, f"ch{ch['number']:02d}_{slug}.mp3")

            # Chunk-level progress for heavy chapters
            def make_cb(i):
                def cb(done, total):
                    jobs[job_id]["chunk_progress"] = f"{done}/{total} chunks"
                return cb

            if not os.path.exists(out_path):
                loop.run_until_complete(
                    tts_chapter(ch["text"], out_path, voice, rate, progress_cb=make_cb(idx))
                )

            jobs[job_id]["chapters"][idx]["done"] = True
            jobs[job_id]["chapters"][idx]["file"] = os.path.basename(out_path)
            jobs[job_id]["progress"]       = idx + 1
            jobs[job_id]["chunk_progress"] = ""

        loop.close()
        jobs[job_id]["status"] = "done"

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"]  = str(e)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    f    = request.files["file"]
    voice = request.form.get("voice", "en-GB-SoniaNeural")
    rate  = request.form.get("rate",  "-10%")

    ext = Path(f.filename).suffix.lower()
    if ext not in (".pdf", ".epub", ".txt"):
        return jsonify({"error": "Only PDF, EPUB, and TXT supported"}), 400

    job_id        = str(uuid.uuid4())[:8]
    book_audio_dir = os.path.join(AUDIO_DIR, job_id)
    os.makedirs(book_audio_dir, exist_ok=True)

    save_path = os.path.join(UPLOAD_DIR, job_id + ext)
    f.save(save_path)

    jobs[job_id] = {
        "status":   "queued",
        "progress": 0,
        "total":    0,
        "chapters": [],
        "error":    None,
        "voice":    voice,
        "rate":     rate,
        "filename": f.filename,
    }

    thread = threading.Thread(
        target=run_job,
        args=(job_id, save_path, voice, rate, book_audio_dir),
        daemon=True
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/audio/<job_id>/<filename>")
def serve_audio(job_id, filename):
    path = os.path.join(AUDIO_DIR, job_id, filename)
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(path, mimetype="audio/mpeg")


@app.route("/voices")
def voices():
    return jsonify([
        {"id": "en-GB-SoniaNeural",   "label": "Sonia (British, Female)"},
        {"id": "en-GB-RyanNeural",    "label": "Ryan (British, Male)"},
        {"id": "en-US-JennyNeural",   "label": "Jenny (American, Female)"},
        {"id": "en-US-GuyNeural",     "label": "Guy (American, Male)"},
        {"id": "en-US-ChristopherNeural", "label": "Christopher (American, Male — deep)"},
        {"id": "en-AU-NatashaNeural", "label": "Natasha (Australian, Female)"},
    ])


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Audiobook Web App")
    print("  Open: http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(debug=False, port=5000)