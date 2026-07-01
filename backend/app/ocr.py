"""
OCR module using baidu/Unlimited-OCR.
Model is loaded once at startup and reused across requests.
"""

import os

# Must be set before torch initializes the CUDA allocator. expandable_segments
# lets PyTorch grow allocations without leaving fragmented, unusable gaps, which
# is what pushes us over the edge on a shared/near-full GPU.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
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

# Pages are OCR'd in batches so peak memory does not scale with page count.
# A 491-page PDF pushed a single infer_multi call to ~12 GB RSS and got the
# process OOM-killed by the kernel; batching bounds that regardless of size.
OCR_BATCH_SIZE = int(os.environ.get("OCR_BATCH_SIZE", "8"))

# Only try to (re)claim the GPU when at least this much VRAM is free, so we
# don't thrash a ~6 GB model on/off the card when Ollama is using it.
MIN_GPU_FREE_BYTES = int(os.environ.get("OCR_MIN_GPU_FREE_GB", "7")) * 1024**3

_model = None
_tokenizer = None
_model_device = "cpu"
_model_lock = threading.Lock()
_model_ready = threading.Event()
_model_error: Optional[str] = None


def _is_oom(exc: Exception) -> bool:
    """True if the exception is a CUDA out-of-memory condition."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def _place_on_gpu() -> bool:
    """
    Try to move the loaded model onto the GPU.
    Returns True if it now lives on the GPU, False if it stayed on the CPU
    (e.g. because the GPU was full — typically Ollama holding VRAM).
    """
    global _model, _model_device

    if not torch.cuda.is_available():
        _model_device = "cpu"
        logger.warning("CUDA not available — running OCR on CPU (expect slow performance)")
        return False

    try:
        torch.cuda.empty_cache()
        _model = _model.cuda()
        torch.cuda.empty_cache()
        _model_device = "cuda"
        logger.info("Model placed on GPU: %s", torch.cuda.get_device_name(0))
        return True
    except Exception as e:
        if not _is_oom(e):
            raise
        # GPU is full (e.g. an Ollama model is loaded). Fall back to CPU rather
        # than crashing; free whatever partial allocation we made first.
        _model = _model.cpu()
        torch.cuda.empty_cache()
        _model_device = "cpu"
        logger.warning(
            "GPU out of memory while placing OCR model — falling back to CPU. "
            "Free VRAM (e.g. unload Ollama models) for faster OCR."
        )
        return False


def _maybe_promote_to_gpu():
    """
    If the model is on the CPU but the GPU now has enough free VRAM (e.g. an
    Ollama model was unloaded), move it back to the GPU. No-op otherwise.
    """
    global _model, _model_device

    if _model_device == "cuda" or not torch.cuda.is_available():
        return

    try:
        free, _total = torch.cuda.mem_get_info()
    except Exception:
        return
    if free < MIN_GPU_FREE_BYTES:
        return

    try:
        torch.cuda.empty_cache()
        _model = _model.cuda()
        torch.cuda.empty_cache()
        _model_device = "cuda"
        logger.info("VRAM free again (%.1f GiB) — moved OCR model back to GPU.", free / 1024**3)
    except Exception as e:
        if not _is_oom(e):
            raise
        _model = _model.cpu()
        torch.cuda.empty_cache()
        _model_device = "cpu"


def load_model():
    """Load the Unlimited-OCR model in a background thread."""
    global _model, _tokenizer, _model_error

    try:
        logger.info("Loading Unlimited-OCR model (this may take a while on first run)...")
        from transformers import AutoModel, AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        # low_cpu_mem_usage streams weights in rather than materializing a second
        # full copy, keeping the load-time memory peak down.
        _model = AutoModel.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        _model = _model.eval()
        _place_on_gpu()

        _model_ready.set()
        logger.info("Unlimited-OCR model ready (device=%s).", _model_device)
    except Exception as e:
        _model_error = str(e)
        # Drop the partially-placed model and free its GPU allocation so a
        # subsequent retry isn't starved by leaked memory.
        _model = None
        _tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
        return {"status": "ready", "device": _model_device}
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

    def _infer(batch: list[str], out_dir: str):
        return _model.infer_multi(
            _tokenizer,
            prompt="<image>Multi page parsing.",
            image_files=batch,
            output_path=out_dir,
            image_size=1024,
            max_length=32768,
            no_repeat_ngram_size=35,
            ngram_window=1024,
            save_results=True,
        )

    def _extract_text(result, out_dir: str) -> str:
        # Prefer the return value, fall back to reading saved output files.
        if isinstance(result, str) and result.strip():
            return result
        if isinstance(result, (list, tuple)):
            joined = "\n\n".join(str(r) for r in result if r)
            if joined.strip():
                return joined
        return _collect_ocr_output(out_dir)

    with _model_lock:
        global _model, _model_device

        # Process in batches so peak memory stays bounded regardless of how many
        # pages the PDF has.
        batches = [
            image_paths[i : i + OCR_BATCH_SIZE]
            for i in range(0, len(image_paths), OCR_BATCH_SIZE)
        ]
        texts: list[str] = []

        for bi, batch in enumerate(batches):
            # Reclaim the GPU between batches if VRAM has freed up (e.g. Ollama
            # unloaded a model). Cheap no-op when already on GPU or still full.
            _maybe_promote_to_gpu()

            if progress_callback:
                start = bi * OCR_BATCH_SIZE + 1
                end = start + len(batch) - 1
                progress_callback(
                    f"OCR pages {start}–{end} of {len(image_paths)} "
                    f"(on {_model_device.upper()})..."
                )

            tmp_out = tempfile.mkdtemp(prefix="ocr_output_")
            try:
                result = _infer(batch, tmp_out)
            except Exception as e:
                # GPU filled up mid-run (e.g. Ollama grabbed VRAM). Drop to CPU
                # and retry this batch rather than failing the whole job.
                if not (_is_oom(e) and _model_device == "cuda"):
                    raise
                logger.warning("GPU OOM during OCR — moving model to CPU and retrying batch.")
                if progress_callback:
                    progress_callback("GPU busy — switching to CPU (slower)...")
                _model = _model.cpu()
                torch.cuda.empty_cache()
                _model_device = "cpu"
                tmp_out = tempfile.mkdtemp(prefix="ocr_output_")
                result = _infer(batch, tmp_out)

            texts.append(_extract_text(result, tmp_out))

            # Free per-batch memory before moving on.
            del result
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return "\n\n".join(t for t in texts if t).strip()
