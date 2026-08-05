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
import time
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

# Unlimited-OCR is GPU-only (its infer code hardcodes .cuda()), and it needs
# ~6.5 GB. We time-share the GPU with Ollama: the model is parked in CPU RAM
# when idle and only moved onto the GPU for the duration of a job, once enough
# VRAM is free. Require this much free VRAM before claiming the card.
MIN_GPU_FREE_BYTES = int(os.environ.get("OCR_MIN_GPU_FREE_GB", "7")) * 1024**3

# How long a job will wait for the GPU to free up (e.g. Ollama to idle/unload)
# before giving up, and how often to re-check while waiting.
GPU_WAIT_TIMEOUT = float(os.environ.get("OCR_GPU_WAIT_TIMEOUT", "900"))
GPU_POLL_INTERVAL = float(os.environ.get("OCR_GPU_POLL_INTERVAL", "10"))

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


def _acquire_gpu(progress_callback=None, timeout: float = GPU_WAIT_TIMEOUT):
    """
    Move the model onto the GPU, waiting for VRAM to free up (e.g. for Ollama to
    idle/unload) if the card is currently too full. Blocks up to `timeout`
    seconds; raises RuntimeError if the GPU never frees in time.

    This model can only run on CUDA, so there is no CPU fallback — we wait.
    """
    global _model, _model_device

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; Unlimited-OCR requires a GPU.")
    if _model_device == "cuda":
        return

    deadline = time.monotonic() + timeout
    warned = False
    while True:
        try:
            free, _total = torch.cuda.mem_get_info()
        except Exception:
            free = 0

        if free >= MIN_GPU_FREE_BYTES:
            try:
                torch.cuda.empty_cache()
                _model = _model.cuda()
                torch.cuda.empty_cache()
                _model_device = "cuda"
                logger.info("Acquired GPU for OCR (%.1f GiB free).", free / 1024**3)
                return
            except Exception as e:
                if not _is_oom(e):
                    raise
                # Lost the race for VRAM; drop back and keep waiting.
                _model = _model.cpu()
                torch.cuda.empty_cache()
                _model_device = "cpu"

        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"GPU still busy after {int(timeout)}s (only {free / 1024**3:.1f} GiB free, "
                f"need {MIN_GPU_FREE_BYTES / 1024**3:.1f}). Free VRAM (e.g. unload Ollama "
                "models) and retry."
            )

        if progress_callback and not warned:
            progress_callback(
                f"Waiting for GPU memory — only {free / 1024**3:.1f} GiB free "
                "(Ollama may be using it)..."
            )
            warned = True
        logger.info("Waiting for GPU VRAM: %.1f GiB free.", free / 1024**3)
        time.sleep(GPU_POLL_INTERVAL)


def _release_gpu():
    """Park the model back in CPU RAM, freeing VRAM for other GPU users (Ollama)."""
    global _model, _model_device
    if _model is not None and _model_device == "cuda":
        _model = _model.cpu()
        _model_device = "cpu"
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model():
    """Load the Unlimited-OCR model in a background thread."""
    global _model, _tokenizer, _model_error, _model_device

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
        # Keep the model parked in CPU RAM at rest so it doesn't hold VRAM while
        # idle — it's moved onto the GPU only for the duration of an OCR job.
        _model = _model.eval()
        _model_device = "cpu"

        _model_ready.set()
        logger.info("Unlimited-OCR model ready (parked in CPU RAM; GPU acquired per job).")
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
        # "active" while a job holds the GPU, "ready" while parked in CPU RAM.
        return {"status": "ready", "gpu_active": _model_device == "cuda"}
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


def pdf_to_markdown(pdf_path: str) -> str:
    """
    Extract a usable Markdown representation from a PDF text layer.

    Text-based PDFs already contain the exact Unicode glyphs used by their
    equations. Using that layer avoids asking OCR to reconstruct subscripts,
    operators, and symbols from a downscaled page image. The function returns
    an empty string for scanned/image-only PDFs so callers can fall back to
    OCR. PDF bookmarks become Markdown headings and non-background embedded
    images become stable references that the EPUB builder can package.
    """
    doc = fitz.open(pdf_path)
    try:
        toc_by_page: dict[int, list[tuple[int, str]]] = {}
        for level, title, page_number in doc.get_toc(simple=True):
            if title.strip() and page_number > 0:
                toc_by_page.setdefault(page_number - 1, []).append(
                    (min(max(level, 1), 2), title.strip())
                )

        chunks: list[str] = []
        nonempty_pages = 0
        for page_index, page in enumerate(doc):
            page_text = page.get_text("text").strip()
            if not page_text:
                continue
            nonempty_pages += 1
            parts = [f"{'#' * level} {title}" for level, title in toc_by_page.get(page_index, [])]
            parts.append(page_text)

            # Keep image references in the extracted Markdown. Skip full-page
            # backgrounds and tiny masks; those are not reader-useful figures.
            seen_xrefs: set[int] = set()
            for image_index, image in enumerate(page.get_images(full=True)):
                xref, _smask, width, height = image[:4]
                if xref in seen_xrefs or width < 80 or height < 80:
                    continue
                seen_xrefs.add(xref)
                rects = page.get_image_rects(xref)
                if rects and any(
                    rect.get_area() >= page.rect.get_area() * 0.8 for rect in rects
                ):
                    continue
                parts.append(f"![](images/page_{page_index}_{image_index}.jpg)")
            chunks.append("\n\n".join(parts))

        # A scanned PDF may have a sparse metadata layer; do not mistake it
        # for usable source text and silently skip OCR.
        if nonempty_pages < max(1, int(len(doc) * 0.60)):
            return ""
        return "\n\n".join(chunks).strip()
    finally:
        doc.close()


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

    # ~4 096 tokens per page is generous for typical book/document content.
    # Capping proportionally prevents the model from filling unused budget
    # with hallucinated text when pages are sparse or short.
    MAX_TOKENS_PER_PAGE = 4096

    def _infer(batch: list[str], out_dir: str):
        batch_max = min(32768, MAX_TOKENS_PER_PAGE * len(batch))
        return _model.infer_multi(
            _tokenizer,
            prompt="<image>Multi page parsing.",
            image_files=batch,
            output_path=out_dir,
            image_size=1024,
            max_length=batch_max,
            no_repeat_ngram_size=35,
            ngram_window=min(1024, batch_max),
            save_results=True,
        )

    def _clean(text: str) -> str:
        """
        Strip LLM artefacts from raw model output before it reaches the EPUB:

        • Markdown code-fence wrappers (```markdown … ``` or ``` … ```)
          Some models wrap their entire output in a fenced block.
        • Common chat/instruction special tokens (<|im_end|>, </s>, <|eot|>, …)
        • <PAGE> page-delimiter markers
        • Leading/trailing whitespace
        """
        import re as _re

        # Strip outer code-fence wrapper produced by some model checkpoints.
        # Matches ``` or ```markdown / ```text at the start, ``` at the end.
        stripped = _re.sub(
            r"^```[a-zA-Z]*\n([\s\S]*?)\n```\s*$",
            r"\1",
            text.strip(),
        )
        # Fall back to the raw text if the regex didn't match (not wrapped).
        text = stripped if stripped != text.strip() else text

        # Remove well-known special tokens produced by various base models.
        for token in (
            "<|im_end|>", "<|im_start|>", "<|endoftext|>",
            "</s>", "<s>", "<|eot_id|>", "<|start_header_id|>",
            "<|end_header_id|>", "<|finetune_right_pad_id|>",
        ):
            text = text.replace(token, "")

        # Unlimited-OCR emits a literal <PAGE> marker between pages. It is
        # not book text, and downstream markdown passes it through as raw
        # unclosed HTML that breaks the XHTML well-formedness of every
        # chapter it lands in.
        text = _re.sub(r"</?PAGE\b[^>\n]*>", "", text)

        return text.strip()

    def _extract_text(result, out_dir: str) -> str:
        # infer_multi may return (markdown_text, token_count) or just the text.
        if isinstance(result, (list, tuple)):
            result = result[0] if result else None
        raw = result if isinstance(result, str) and result.strip() else _collect_ocr_output(out_dir)
        cleaned = _clean(raw) if raw else ""
        if cleaned:
            logger.debug("OCR batch output: %d chars — preview: %.200s", len(cleaned), cleaned)
        return cleaned

    with _model_lock:
        # Grab the GPU for this job, waiting for VRAM if Ollama is using it.
        _acquire_gpu(progress_callback)
        try:
            # Process in batches so peak memory stays bounded regardless of how
            # many pages the PDF has.
            batches = [
                image_paths[i : i + OCR_BATCH_SIZE]
                for i in range(0, len(image_paths), OCR_BATCH_SIZE)
            ]
            texts: list[str] = []

            for bi, batch in enumerate(batches):
                if progress_callback:
                    start = bi * OCR_BATCH_SIZE + 1
                    end = start + len(batch) - 1
                    progress_callback(
                        f"OCR pages {start}–{end} of {len(image_paths)}..."
                    )

                tmp_out = tempfile.mkdtemp(prefix="ocr_output_")
                try:
                    result = _infer(batch, tmp_out)
                except Exception as e:
                    # VRAM ran out mid-job (e.g. Ollama grabbed the card). Free
                    # the GPU, wait for room, re-acquire, and retry this batch
                    # rather than failing the whole job.
                    if not _is_oom(e):
                        raise
                    logger.warning("GPU OOM during OCR batch — releasing and waiting for VRAM.")
                    _release_gpu()
                    _acquire_gpu(progress_callback)
                    tmp_out = tempfile.mkdtemp(prefix="ocr_output_")
                    result = _infer(batch, tmp_out)

                texts.append(_extract_text(result, tmp_out))

                # Free per-batch memory before moving on.
                del result
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            return "\n\n".join(t for t in texts if t).strip()
        finally:
            # Always hand the GPU back so Ollama (and the next job) can use it.
            _release_gpu()
