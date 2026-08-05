"""
pdf2epub backend — FastAPI application.

Endpoints:
  POST /api/upload         Upload a PDF; returns job_id and page_count
  GET  /api/thumbnail/{job_id}/{page}  Page thumbnail (JPEG)
  POST /api/process        Start OCR + EPUB conversion
  GET  /api/status/{job_id}           Poll job status
  GET  /api/model-status   Check model loading status
  GET  /api/library                   List EPUBs in the output directory
  GET  /api/library/{filename}/download  Download an EPUB from the output directory
  POST /api/library/{filename}/validate  Quick structural check of an EPUB
  POST /api/library/{filename}/validate-full  Full validation: spec (EPUBCheck),
                                      simulated Chromium renders, and — when a
                                      source PDF is attached — fidelity comparison
  GET  /api/library/{filename}/validation-report  Last stored full report
  GET  /api/library/{filename}/screenshots/{shot}  Render screenshot (PNG)
  POST /api/library/{filename}/fix       Attempt to repair a broken EPUB
"""

import os
import io
import json
import uuid
import shutil
import asyncio
import logging
import tempfile
import threading
from pathlib import Path
from typing import Literal, Optional

import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse, FileResponse
from pydantic import BaseModel

from .ocr import (
    start_model_loading,
    run_ocr,
    pdf_to_markdown,
    pdf_to_images,
    render_page_thumbnail,
    render_page_image,
    get_model_status,
    is_model_ready,
)
from .epub_builder import (
    build_epub,
    extract_pdf_images_for_markdown,
    validate_epub_report,
    repair_epub,
)
from .epub_validator import validate_full

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
    mode: Literal["reader_safe", "full"] = "reader_safe"


def _run_job(
    job_id: str,
    cover_page: Optional[int],
    output_filename: str,
    mode: Literal["reader_safe", "full"],
):
    """Background thread: OCR → EPUB."""
    try:
        job = _get_job(job_id)
        pdf_path = job["pdf_path"]
        job_dir = _job_dir(job_id)

        # ── 1. Prefer the PDF text layer when it exists ──
        # This preserves the source document's Unicode equation glyphs and
        # avoids spending GPU time OCR'ing a text-based PDF. Scanned PDFs
        # return an empty string and continue through the OCR path.
        source_markdown = pdf_to_markdown(pdf_path)
        if source_markdown:
            image_paths = []
            _update_job(job_id, status="processing", step="Using embedded PDF text layer...")
            logger.info("[%s] Using embedded PDF text layer", job_id)
        else:
            _update_job(job_id, status="processing", step="Rendering PDF pages to images...")
            logger.info("[%s] Rendering pages", job_id)
            image_paths = pdf_to_images(pdf_path, dpi=300)
            _update_job(job_id, step=f"Rendered {len(image_paths)} pages. Starting OCR...")

        # ── 2. Extract cover image if requested ──
        cover_bytes = None
        if mode == "full" and cover_page is not None and 0 <= cover_page < job["page_count"]:
            _update_job(job_id, step=f"Extracting cover from page {cover_page + 1}...")
            cover_bytes = render_page_image(pdf_path, cover_page, dpi=150)

        # ── 3. Extract text ──
        if source_markdown:
            ocr_text = source_markdown
        else:
            _update_job(job_id, step="Running OCR (this may take several minutes)...")
            logger.info("[%s] Starting OCR on %d pages", job_id, len(image_paths))

            def progress(msg):
                _update_job(job_id, step=msg)

            ocr_text = run_ocr(image_paths, progress_callback=progress)

        if not ocr_text.strip():
            raise RuntimeError("OCR returned no text. Check that the PDF is legible.")

        if mode == "reader_safe":
            _update_job(
                job_id,
                step=f"Text extraction complete ({len(ocr_text)} chars). Preparing reader-safe EPUB...",
            )
            images = {}
        else:
            _update_job(job_id, step=f"OCR complete ({len(ocr_text)} chars). Extracting figures...")

            # ── 4. Pull the images the OCR markdown references out of the PDF ──
            # so the EPUB packages real figures instead of broken <img> tags.
            images = extract_pdf_images_for_markdown(pdf_path, ocr_text)

        _update_job(job_id, step="Building EPUB...")

        # ── 5. Build EPUB ──
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
            images=images,
            mode=mode,
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

    if not is_model_ready() and not pdf_to_markdown(job["pdf_path"]):
        raise HTTPException(
            status_code=503,
            detail="Model is still loading. Please wait and try again.",
        )

    output_filename = req.output_filename or Path(job.get("original_name", "document.pdf")).stem

    # Run in a real thread so we don't block the async event loop
    t = threading.Thread(
        target=_run_job,
        args=(req.job_id, req.cover_page, output_filename, req.mode),
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


# ── Library (validate/fix existing output EPUBs) ─────────────────────────────

def _library_path(filename: str) -> Path:
    # Path(filename).name strips any directory components, so a mismatch
    # against the original means path traversal (e.g. "../../etc/passwd")
    # was attempted.
    safe_name = Path(filename).name
    if not safe_name or safe_name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    path = OUTPUT_DIR / safe_name
    if path.suffix.lower() != ".epub" or not path.is_file():
        raise HTTPException(status_code=404, detail="EPUB not found")
    return path


@app.get("/api/library")
async def list_library():
    files = []
    for p in sorted(OUTPUT_DIR.glob("*.epub")):
        st = p.stat()
        files.append({"filename": p.name, "size": st.st_size, "modified": st.st_mtime})
    return {"files": files}


@app.get("/api/library/{filename}/download")
async def download_library_epub(filename: str):
    path = _library_path(filename)
    return FileResponse(path, media_type="application/epub+zip", filename=path.name)


@app.post("/api/library/{filename}/validate")
async def validate_library_epub(filename: str):
    path = _library_path(filename)
    issues = validate_epub_report(str(path))
    return {"filename": path.name, "valid": not issues, "issues": issues}


def _validation_dir(epub_path: Path) -> Path:
    return OUTPUT_DIR / ".validation" / epub_path.stem


def _find_source_pdf(filename: str) -> Optional[str]:
    """If the job that produced this EPUB is still around, reuse its input PDF."""
    with JOBS_LOCK:
        for job in JOBS.values():
            if job.get("output_file") == filename and os.path.isfile(job.get("pdf_path", "")):
                return job["pdf_path"]
    return None


@app.post("/api/library/{filename}/validate-full")
async def validate_library_epub_full(
    filename: str,
    pdf: Optional[UploadFile] = File(None),
    render: bool = True,
    epubcheck: bool = True,
    screenshots: bool = True,
):
    """
    Full validation of an EPUB in the library:

      * format  — OCF/OPF/content checks + the official W3C EPUBCheck
      * render  — every spine document opened in headless Chromium
      * fidelity — text/TOC/image comparison against the source PDF, when one
        is attached as multipart field 'pdf' (or the original upload is still
        available from the conversion job)

    The JSON report is returned and also persisted next to the library so
    GET /validation-report can serve it later.
    """
    path = _library_path(filename)

    pdf_path = _find_source_pdf(filename)
    tmp_pdf = None
    if pdf is not None:
        if not (pdf.filename or "").lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Comparison file must be a PDF")
        contents = await pdf.read()
        tmp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp_pdf.write(contents)
        tmp_pdf.close()
        pdf_path = tmp_pdf.name

    report_dir = _validation_dir(path)
    shots_dir = None
    if render and screenshots:
        shots_dir = report_dir / "screenshots"
        shutil.rmtree(shots_dir, ignore_errors=True)

    try:
        report = await asyncio.to_thread(
            validate_full,
            str(path),
            pdf_path=pdf_path,
            render=render,
            epubcheck=epubcheck,
            screenshots_dir=str(shots_dir) if shots_dir else None,
        )
    finally:
        if tmp_pdf is not None:
            os.unlink(tmp_pdf.name)

    # Screenshots as API-servable names rather than server paths.
    for doc in report.get("render", {}).get("documents", []):
        if doc.get("screenshot"):
            doc["screenshot"] = os.path.basename(doc["screenshot"])
    report["epub"] = path.name
    report["pdf"] = bool(pdf_path)

    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "report.json", "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2, ensure_ascii=False)

    return report


@app.get("/api/library/{filename}/validation-report")
async def get_validation_report(filename: str):
    path = _library_path(filename)
    report_file = _validation_dir(path) / "report.json"
    if not report_file.is_file():
        raise HTTPException(status_code=404, detail="No stored validation report; run validate-full first")
    with open(report_file, "r", encoding="utf-8") as fp:
        return json.load(fp)


@app.get("/api/library/{filename}/screenshots/{shot}")
async def get_validation_screenshot(filename: str, shot: str):
    path = _library_path(filename)
    safe_shot = Path(shot).name
    if safe_shot != shot or not safe_shot.endswith(".png"):
        raise HTTPException(status_code=400, detail="Invalid screenshot name")
    shot_path = _validation_dir(path) / "screenshots" / safe_shot
    if not shot_path.is_file():
        raise HTTPException(status_code=404, detail="Screenshot not found")
    return Response(content=shot_path.read_bytes(), media_type="image/png")


@app.post("/api/library/{filename}/fix")
async def fix_library_epub(filename: str):
    path = _library_path(filename)
    result = repair_epub(str(path))
    return {"filename": path.name, **result}
