"""
pdf2epub backend — FastAPI application.

Endpoints:
  POST /api/upload         Upload a PDF; returns job_id and page_count
  GET  /api/thumbnail/{job_id}/{page}  Page thumbnail (JPEG)
  POST /api/process        Start OCR + EPUB conversion
  GET  /api/status/{job_id}           Poll job status
  GET  /api/model-status   Check model loading status
"""

import os
import io
import uuid
import shutil
import asyncio
import logging
import tempfile
import threading
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel

from .ocr import (
    start_model_loading,
    run_ocr,
    pdf_to_images,
    render_page_thumbnail,
    render_page_image,
    get_model_status,
    is_model_ready,
)
from .epub_builder import build_epub

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TMP_ROOT = Path("/tmp/pdf2epub")
TMP_ROOT.mkdir(parents=True, exist_ok=True)

# ── In-memory job store ──────────────────────────────────────────────────────
# job_id -> {
#   status: "pending"|"processing"|"done"|"error"
#   step: str          (current human-readable step)
#   pdf_path: str
#   page_count: int
#   output_file: str | None
#   error: str | None
# }
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="pdf2epub")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    start_model_loading()
    logger.info("Model loading started in background.")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _job_dir(job_id: str) -> Path:
    d = TMP_ROOT / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get_job(job_id: str) -> dict:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _update_job(job_id: str, **kwargs):
    with JOBS_LOCK:
        JOBS[job_id].update(kwargs)


# ── Upload ────────────────────────────────────────────────────────────────────

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    job_id = str(uuid.uuid4())
    job_dir = _job_dir(job_id)
    pdf_path = str(job_dir / "input.pdf")

    # Save uploaded file
    contents = await file.read()
    with open(pdf_path, "wb") as f:
        f.write(contents)

    # Count pages
    try:
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        doc.close()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not open PDF: {e}")

    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "pending",
            "step": "Uploaded — waiting to process",
            "pdf_path": pdf_path,
            "page_count": page_count,
            "original_name": file.filename,
            "output_file": None,
            "error": None,
        }

    return {"job_id": job_id, "page_count": page_count, "filename": file.filename}


# ── Thumbnails ────────────────────────────────────────────────────────────────

@app.get("/api/thumbnail/{job_id}/{page}")
async def get_thumbnail(job_id: str, page: int):
    job = _get_job(job_id)
    page_count = job["page_count"]

    if page < 0 or page >= page_count:
        raise HTTPException(status_code=400, detail=f"Page {page} out of range (0–{page_count - 1})")

    try:
        jpeg_bytes = render_page_thumbnail(job["pdf_path"], page, max_dim=300)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return Response(content=jpeg_bytes, media_type="image/jpeg")


# ── Process ───────────────────────────────────────────────────────────────────

class ProcessRequest(BaseModel):
    job_id: str
    cover_page: Optional[int] = None   # 0-indexed page to use as cover; None = no cover
    output_filename: Optional[str] = None  # base name without extension


def _run_job(job_id: str, cover_page: Optional[int], output_filename: str):
    """Background thread: OCR → EPUB."""
    try:
        job = _get_job(job_id)
        pdf_path = job["pdf_path"]
        job_dir = _job_dir(job_id)

        # ── 1. Render PDF pages to images ──
        _update_job(job_id, status="processing", step="Rendering PDF pages to images...")
        logger.info("[%s] Rendering pages", job_id)
        image_paths = pdf_to_images(pdf_path, dpi=300)
        _update_job(job_id, step=f"Rendered {len(image_paths)} pages. Starting OCR...")

        # ── 2. Extract cover image if requested ──
        cover_bytes = None
        if cover_page is not None and 0 <= cover_page < job["page_count"]:
            _update_job(job_id, step=f"Extracting cover from page {cover_page + 1}...")
            cover_bytes = render_page_image(pdf_path, cover_page, dpi=150)

        # ── 3. Run OCR ──
        _update_job(job_id, step="Running OCR (this may take several minutes)...")
        logger.info("[%s] Starting OCR on %d pages", job_id, len(image_paths))

        def progress(msg):
            _update_job(job_id, step=msg)

        ocr_text = run_ocr(image_paths, progress_callback=progress)

        if not ocr_text.strip():
            raise RuntimeError("OCR returned no text. Check that the PDF is legible.")

        _update_job(job_id, step=f"OCR complete ({len(ocr_text)} chars). Building EPUB...")

        # ── 4. Build EPUB ──
        original_name = job.get("original_name", "document.pdf")
        title = output_filename or Path(original_name).stem
        safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in title).strip()
        epub_filename = f"{safe_name}.epub"
        epub_path = str(OUTPUT_DIR / epub_filename)

        build_epub(
            title=title,
            author="pdf2epub",
            ocr_text=ocr_text,
            output_path=epub_path,
            cover_image_bytes=cover_bytes,
            cover_image_mime="image/png",
        )

        _update_job(
            job_id,
            status="done",
            step="Done",
            output_file=epub_filename,
        )
        logger.info("[%s] Done → %s", job_id, epub_path)

    except Exception as e:
        logger.exception("[%s] Job failed: %s", job_id, e)
        _update_job(job_id, status="error", step="Failed", error=str(e))


@app.post("/api/process")
async def process(req: ProcessRequest):
    job = _get_job(req.job_id)

    if job["status"] == "processing":
        raise HTTPException(status_code=409, detail="Job is already processing")
    if job["status"] == "done":
        raise HTTPException(status_code=409, detail="Job already completed")

    if not is_model_ready():
        raise HTTPException(
            status_code=503,
            detail="Model is still loading. Please wait and try again.",
        )

    output_filename = req.output_filename or Path(job.get("original_name", "document.pdf")).stem

    # Run in a real thread so we don't block the async event loop
    t = threading.Thread(
        target=_run_job,
        args=(req.job_id, req.cover_page, output_filename),
        daemon=True,
    )
    t.start()

    return {"job_id": req.job_id, "status": "processing"}


# ── Status ────────────────────────────────────────────────────────────────────

@app.get("/api/status/{job_id}")
async def status(job_id: str):
    job = _get_job(job_id)
    return {
        "job_id": job_id,
        "status": job["status"],
        "step": job["step"],
        "page_count": job["page_count"],
        "output_file": job.get("output_file"),
        "error": job.get("error"),
    }


@app.get("/api/model-status")
async def model_status():
    return get_model_status()
