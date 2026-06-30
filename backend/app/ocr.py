"""
OCR module using baidu/Unlimited-OCR.
Model is loaded once at startup and reused across requests.
"""

import os
import glob
import tempfile
import threading
import logging
from pathlib import Path
from typing import Optional

import torch
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

MODEL_NAME = "baidu/Unlimited-OCR"

_model = None
_tokenizer = None
_model_lock = threading.Lock()
_model_ready = threading.Event()
_model_error: Optional[str] = None


def load_model():
    """Load the Unlimited-OCR model in a background thread."""
    global _model, _tokenizer, _model_error

    try:
        logger.info("Loading Unlimited-OCR model (this may take a while on first run)...")
        from transformers import AutoModel, AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        _model = AutoModel.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
        )

        if torch.cuda.is_available():
            _model = _model.eval().cuda()
            logger.info("Model loaded on GPU: %s", torch.cuda.get_device_name(0))
        else:
            _model = _model.eval()
            logger.warning("CUDA not available — running on CPU (expect slow performance)")

        _model_ready.set()
        logger.info("Unlimited-OCR model ready.")
    except Exception as e:
        _model_error = str(e)
        _model_ready.set()  # unblock waiters so they can surface the error
        logger.exception("Failed to load model: %s", e)


def start_model_loading():
    """Kick off model loading in the background."""
    t = threading.Thread(target=load_model, daemon=True)
    t.start()


def wait_for_model(timeout: float = 600.0):
    """Block until model is ready or raise RuntimeError."""
    _model_ready.wait(timeout=timeout)
    if _model_error:
        raise RuntimeError(f"Model failed to load: {_model_error}")
    if _model is None:
        raise RuntimeError("Model not loaded (timeout?)")


def is_model_ready() -> bool:
    return _model_ready.is_set() and _model_error is None


def get_model_status() -> dict:
    if _model_error:
        return {"status": "error", "detail": _model_error}
    if _model_ready.is_set():
        return {"status": "ready"}
    return {"status": "loading"}


def pdf_to_images(pdf_path: str, dpi: int = 300) -> list[str]:
    """Convert each page of a PDF to a PNG and return the file paths."""
    doc = fitz.open(pdf_path)
    tmp_dir = tempfile.mkdtemp(prefix="pdf_ocr_pages_")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    paths = []
    for i, page in enumerate(doc):
        out = os.path.join(tmp_dir, f"page_{i + 1:04d}.png")
        page.get_pixmap(matrix=mat).save(out)
        paths.append(out)
    doc.close()
    return paths


def render_page_thumbnail(pdf_path: str, page_index: int, max_dim: int = 300) -> bytes:
    """Render a single PDF page as a JPEG thumbnail, returned as bytes."""
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    # Scale so the longest side is max_dim
    rect = page.rect
    scale = max_dim / max(rect.width, rect.height)
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat)
    data = pix.tobytes("jpeg")
    doc.close()
    return data


def render_page_image(pdf_path: str, page_index: int, dpi: int = 300) -> bytes:
    """Render a single PDF page at full resolution as PNG bytes."""
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    data = pix.tobytes("png")
    doc.close()
    return data


def _collect_ocr_output(output_dir: str) -> str:
    """
    Read OCR result files written by infer_multi to output_dir.
    Supports .md, .txt, and .json output formats.
    Returns concatenated text.
    """
    texts = []

    # Try markdown first (most structured), then txt, then json
    for pattern in ["*.md", "*.txt"]:
        files = sorted(glob.glob(os.path.join(output_dir, pattern)))
        if files:
            for f in files:
                with open(f, encoding="utf-8") as fh:
                    texts.append(fh.read())
            return "\n\n".join(texts)

    # JSON fallback
    import json
    for f in sorted(glob.glob(os.path.join(output_dir, "*.json"))):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, str):
            texts.append(data)
        elif isinstance(data, dict):
            texts.append(data.get("text", data.get("content", str(data))))
        elif isinstance(data, list):
            texts.extend(str(item) for item in data)

    return "\n\n".join(texts)


def run_ocr(image_paths: list[str], progress_callback=None) -> str:
    """
    Run Unlimited-OCR on a list of page images.
    Returns the concatenated markdown/text output.
    """
    wait_for_model()

    with _model_lock:
        tmp_out = tempfile.mkdtemp(prefix="ocr_output_")

        if progress_callback:
            progress_callback("Running OCR on all pages...")

        result = _model.infer_multi(
            _tokenizer,
            prompt="<image>Multi page parsing.",
            image_files=image_paths,
            output_path=tmp_out,
            image_size=1024,
            max_length=32768,
            no_repeat_ngram_size=35,
            ngram_window=1024,
            save_results=True,
        )

        # Try to get text from return value first
        text = None
        if isinstance(result, str) and result.strip():
            text = result
        elif isinstance(result, (list, tuple)):
            text = "\n\n".join(str(r) for r in result if r)

        # Fall back to reading saved output files
        if not text:
            text = _collect_ocr_output(tmp_out)

        return text or ""
