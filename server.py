"""
Audiobook Web App — Backend Server
====================================
Run:  python server.py
Then open: http://localhost:5000

Install:
    pip install flask edge-tts pdfplumber ebooklib beautifulsoup4

Features:
- Resume interrupted conversions (chunk-level, not chapter-level)
- Auto-recovers incomplete jobs on server restart
- /books endpoint so frontend can restore previous sessions
- Last position saved server-side per book
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

UPLOAD_DIR = "uploads"
AUDIO_DIR  = "audio"
STATE_FILE = "books_state.json"   # persists job metadata across restarts

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR,  exist_ok=True)

# ── Persistent state ──────────────────────────────────────────────────────────
# Loaded from disk on startup, saved whenever a chapter completes.

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

# In-memory jobs dict — merged with persisted state on startup
jobs = load_state()


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
    if ext == ".pdf":   return extract_from_pdf(filepath)
    if ext == ".epub":  return extract_from_epub(filepath)
    if ext == ".txt":   return extract_from_txt(filepath)
    raise ValueError(f"Unsupported format: {ext}")


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text):
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*(www\.|http)\S+\s*$', '', text, flags=re.MULTILINE)
    return text.strip()


# ── Chapter splitting ─────────────────────────────────────────────────────────

CHAPTER_NUM_RE = re.compile(
    r'(?:^|\n)[ \t]*C\s*H\s*A\s*P\s*T\s*E\s*R[ \t]+(\d+)[ \t]*(?:\r?\n)',
    re.IGNORECASE
)
CHAPTER_INLINE_RE = re.compile(
    r'(?:^|\n)[ \t]*Chapter[ \t]+(\d+)[ \t]*[:–-][ \t]*([^\n]{3,80})',
    re.IGNORECASE
)

def slugify(text):
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'\s+', '_', text.strip())
    return text[:50]

def _grab_title_after(full_text, pos_after_num_line):
    segment = full_text[pos_after_num_line: pos_after_num_line + 300]
    for line in segment.split('\n')[:6]:
        s = line.strip()
        if not s: continue
        if re.match(r'^[_\-=\s]+$', s): continue
        if re.match(r'^\d+\.', s): continue
        if len(s) < 3: continue
        return s
    return ""

def find_chapter_positions(full_text):
    candidates = []
    for m in CHAPTER_NUM_RE.finditer(full_text):
        ch_num = int(m.group(1))
        title  = _grab_title_after(full_text, m.end())
        title  = re.sub(r'[\s_\.]+\d*\s*$', '', title).strip() or f"Chapter {ch_num}"
        candidates.append((m.start(), ch_num, title, 'twoline'))
    if not candidates:
        for m in CHAPTER_INLINE_RE.finditer(full_text):
            ch_num = int(m.group(1))
            title  = re.sub(r'[\s_\.]+\d*\s*$', '', m.group(2).strip())
            candidates.append((m.start(), ch_num, title, 'inline'))
    if not candidates:
        return []
    candidates.sort(key=lambda x: x[0])
    seen = {}
    for pos, num, title, fmt in candidates:
        seen[num] = (pos, num, title)
    return sorted(seen.values(), key=lambda x: x[0])

def split_into_chapters(full_text):
    positions = find_chapter_positions(full_text)
    if not positions:
        return []
    chapters = []
    for idx, (pos, ch_num, title) in enumerate(positions):
        next_pos   = positions[idx + 1][0] if idx + 1 < len(positions) else len(full_text)
        title_pos  = full_text.find(title, pos)
        body_start = full_text.find('\n', title_pos) + 1 if title_pos != -1 and title_pos < next_pos else full_text.find('\n', pos) + 1
        body       = clean_text(full_text[body_start:next_pos])
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
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current, current_len = [], [], 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
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
    Convert one chapter to MP3.
    RESUME LOGIC: if a chunk file already exists on disk, skip it entirely.
    This means a restart picks up exactly where it left off, chunk by chunk.
    """
    chunks   = chunk_text(text)
    total    = len(chunks)
    temp_dir = out_path + "_tmp"
    os.makedirs(temp_dir, exist_ok=True)

    semaphore = asyncio.Semaphore(concurrency)
    completed = [0]

    async def run(i, chunk):
        async with semaphore:
            chunk_path = os.path.join(temp_dir, f"chunk_{i:05d}.mp3")
            if os.path.exists(chunk_path):
                # Already done — count it but skip conversion
                completed[0] += 1
                if progress_cb:
                    progress_cb(completed[0], total)
                return
            await tts_chunk(chunk, chunk_path, voice, rate)
            completed[0] += 1
            if progress_cb:
                progress_cb(completed[0], total)

    await asyncio.gather(*[run(i, c) for i, c in enumerate(chunks)])

    # Stitch all chunks into final MP3
    chunk_files = sorted(glob.glob(os.path.join(temp_dir, "chunk_*.mp3")))
    stitch(chunk_files, out_path)
    shutil.rmtree(temp_dir)


def stitch(chunk_files, out_path):
    """Concatenate MP3 bytes directly — no ffmpeg or pydub needed."""
    with open(out_path, "wb") as out:
        for f in chunk_files:
            with open(f, "rb") as chunk:
                out.write(chunk.read())


# ── Background job ────────────────────────────────────────────────────────────

def run_job(job_id, filepath, voice, rate, book_audio_dir):
    try:
        jobs[job_id]["status"] = "extracting"
        save_state(jobs)

        full_text = extract_text(filepath)
        chapters  = split_into_chapters(full_text)

        if not chapters:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"]  = "No chapters detected. Check the PDF has 'CHAPTER N' headings."
            save_state(jobs)
            return

        jobs[job_id]["total"]  = len(chapters)
        jobs[job_id]["status"] = "converting"

        # Merge with any existing chapter state (from a previous interrupted run)
        existing = {c["number"]: c for c in jobs[job_id].get("chapters", [])}
        jobs[job_id]["chapters"] = []
        for ch in chapters:
            prev = existing.get(ch["number"], {})
            jobs[job_id]["chapters"].append({
                "number": ch["number"],
                "title":  ch["title"],
                "words":  ch["words"],
                "done":   prev.get("done", False),
                "file":   prev.get("file", ""),
            })
        save_state(jobs)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        for idx, ch in enumerate(chapters):
            slug     = slugify(ch["title"])
            out_path = os.path.join(book_audio_dir, f"ch{ch['number']:02d}_{slug}.mp3")

            # Skip fully completed chapters
            if jobs[job_id]["chapters"][idx].get("done") and os.path.exists(out_path):
                jobs[job_id]["progress"] = idx + 1
                continue

            # Resume incomplete chapter — tts_chapter skips existing chunks internally
            def make_cb(i):
                def cb(done, total):
                    jobs[job_id]["chunk_progress"] = f"{done}/{total} chunks"
                return cb

            loop.run_until_complete(
                tts_chapter(ch["text"], out_path, voice, rate, progress_cb=make_cb(idx))
            )

            jobs[job_id]["chapters"][idx]["done"] = True
            jobs[job_id]["chapters"][idx]["file"] = os.path.basename(out_path)
            jobs[job_id]["progress"]       = idx + 1
            jobs[job_id]["chunk_progress"] = ""
            save_state(jobs)   # ← persist after every completed chapter

        loop.close()
        jobs[job_id]["status"] = "done"
        save_state(jobs)

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"]  = str(e)
        save_state(jobs)


# ── Auto-recover interrupted jobs on startup ──────────────────────────────────

def recover_interrupted_jobs():
    """
    On server start, find any jobs that were mid-conversion when the server
    stopped, and resume them automatically in background threads.
    """
    for job_id, job in list(jobs.items()):
        if job.get("status") in ("converting", "extracting", "queued"):
            print(f"  Resuming interrupted job: {job_id} ({job.get('filename', '?')})")
            # Find the uploaded file
            upload_path = None
            for ext in (".pdf", ".epub", ".txt"):
                p = os.path.join(UPLOAD_DIR, job_id + ext)
                if os.path.exists(p):
                    upload_path = p
                    break
            if not upload_path:
                jobs[job_id]["status"] = "error"
                jobs[job_id]["error"]  = "Original file not found — please re-upload."
                save_state(jobs)
                continue
            book_audio_dir = os.path.join(AUDIO_DIR, job_id)
            os.makedirs(book_audio_dir, exist_ok=True)
            thread = threading.Thread(
                target=run_job,
                args=(job_id, upload_path, job.get("voice", "en-GB-SoniaNeural"),
                      job.get("rate", "-10%"), book_audio_dir),
                daemon=True
            )
            thread.start()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    f     = request.files["file"]
    voice = request.form.get("voice", "en-GB-SoniaNeural")
    rate  = request.form.get("rate",  "-10%")

    ext = Path(f.filename).suffix.lower()
    if ext not in (".pdf", ".epub", ".txt"):
        return jsonify({"error": "Only PDF, EPUB, and TXT supported"}), 400

    job_id         = str(uuid.uuid4())[:8]
    book_audio_dir = os.path.join(AUDIO_DIR, job_id)
    os.makedirs(book_audio_dir, exist_ok=True)

    save_path = os.path.join(UPLOAD_DIR, job_id + ext)
    f.save(save_path)

    jobs[job_id] = {
        "status":         "queued",
        "progress":       0,
        "total":          0,
        "chapters":       [],
        "error":          None,
        "voice":          voice,
        "rate":           rate,
        "filename":       f.filename,
        "chunk_progress": "",
        "last_position":  {"chapter_idx": 0, "time": 0},
    }
    save_state(jobs)

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


@app.route("/books")
def list_books():
    """Return all known books — used by frontend to restore previous sessions."""
    result = []
    for job_id, job in jobs.items():
        result.append({
            "job_id":        job_id,
            "filename":      job.get("filename", "Unknown"),
            "status":        job.get("status", "unknown"),
            "total":         job.get("total", 0),
            "progress":      job.get("progress", 0),
            "last_position": job.get("last_position", {"chapter_idx": 0, "time": 0}),
            "chapters":      job.get("chapters", []),
        })
    # Most recently added first
    result.reverse()
    return jsonify(result)


@app.route("/position/<job_id>", methods=["POST"])
def save_position(job_id):
    """Save the listener's last playback position for a book."""
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    data = request.json or {}
    jobs[job_id]["last_position"] = {
        "chapter_idx": data.get("chapter_idx", 0),
        "time":        data.get("time", 0),
    }
    save_state(jobs)
    return jsonify({"ok": True})


@app.route("/audio/<job_id>/<filename>")
def serve_audio(job_id, filename):
    path = os.path.join(AUDIO_DIR, job_id, filename)
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(path, mimetype="audio/mpeg")


@app.route("/voices")
def voices():
    return jsonify([
        {"id": "en-GB-SoniaNeural",       "label": "Sonia (British, Female)"},
        {"id": "en-GB-RyanNeural",         "label": "Ryan (British, Male)"},
        {"id": "en-US-JennyNeural",        "label": "Jenny (American, Female)"},
        {"id": "en-US-GuyNeural",          "label": "Guy (American, Male)"},
        {"id": "en-US-ChristopherNeural",  "label": "Christopher (American, Male — deep)"},
        {"id": "en-AU-NatashaNeural",      "label": "Natasha (Australian, Female)"},
    ])


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Audiobook Web App")
    print("  Open: http://localhost:5000")
    print("=" * 50)
    recover_interrupted_jobs()
    print()
    app.run(debug=False, port=5000)