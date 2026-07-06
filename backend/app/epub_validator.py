"""
Full-stack EPUB validation.

Three independent layers, aggregated by validate_full():

  1. Format     — EPUB/OCF spec conformance: container structure, OPF
                  manifest/spine cross-checks, internal reference resolution,
                  plus the official W3C EPUBCheck validator when Java and the
                  epubcheck jar are available (RSC-005 schema errors, etc.).
  2. Render     — simulated renders of every spine document in headless
                  Chromium (Playwright): broken images, failed resource
                  loads, page errors, blank chapters, layout overflow.
  3. Fidelity   — comparison against the source PDF: text coverage (what
                  fraction of the PDF's text survived into the EPUB, and
                  where the gaps are), extra/hallucinated content, heading &
                  TOC coverage, image counts, mojibake detection.

Layers degrade gracefully: if Java/EPUBCheck or Playwright/Chromium are not
installed the corresponding layer is reported as "skipped" rather than
failing the whole validation, so the pure-Python checks always run.

Every finding is a dict:
    {"level": "error"|"warning"|"info", "code": str, "message": str,
     "location": str|None}
and the overall verdict is "fail" (any error), "warn" (warnings only),
or "pass".

CLI:  python -m app.epub_validator book.epub [source.pdf] [--json out.json]
"""

import io
import json
import logging
import os
import posixpath
import re
import shutil
import subprocess
import tempfile
import unicodedata
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse

from lxml import etree
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

OPF_NS = "http://www.idpf.org/2007/opf"
CONTAINER_NS = "urn:oasis:names:tc:opendocument:xmlns:container"
XHTML_NS = "http://www.w3.org/1999/xhtml"
DC_NS = "http://purl.org/dc/elements/1.1/"
EPUB_MIMETYPE = b"application/epub+zip"

# Viewport that approximates a mid-size e-reader screen.
RENDER_VIEWPORT = {"width": 600, "height": 800}

# Fallback Chromium executables for when the Playwright pip package and the
# browsers on disk are different revisions (common in prebuilt containers).
CHROMIUM_FALLBACKS = (
    os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE", ""),
    "/opt/pw-browsers/chromium",
)

# Text-coverage thresholds for the PDF fidelity verdict.
COVERAGE_ERROR_BELOW = 0.75
COVERAGE_WARN_BELOW = 0.90


def _finding(level: str, code: str, message: str, location: str | None = None) -> dict:
    return {"level": level, "code": code, "message": message, "location": location}


# ═════════════════════════════════ Layer 1: format ═══════════════════════════


def _find_epubcheck_jar() -> str | None:
    """Locate an epubcheck jar: $EPUBCHECK_JAR, then the PyPI package's copy."""
    env = os.environ.get("EPUBCHECK_JAR")
    if env and os.path.isfile(env):
        return env
    try:
        import epubcheck as _ec  # PyPI package bundling the official jar

        jar = os.path.join(os.path.dirname(_ec.__file__), "epubcheck.jar")
        if os.path.isfile(jar):
            return jar
    except ImportError:
        pass
    return None


def run_epubcheck(epub_path: str, timeout: int = 600) -> dict:
    """
    Run the official W3C EPUBCheck validator, the reference implementation of
    the EPUB spec. Returns {"available": bool, "findings": [...], "counts":
    {...}} — findings use our normalized shape. Skipped (available=False) when
    Java or the jar is missing.
    """
    jar = _find_epubcheck_jar()
    java = shutil.which("java")
    if not jar or not java:
        return {
            "available": False,
            "reason": "java or epubcheck.jar not found "
            "(pip install epubcheck; apt install default-jre-headless)",
            "findings": [],
        }

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        report_path = tf.name
    try:
        proc = subprocess.run(
            [java, "-jar", jar, "--json", report_path, epub_path],
            capture_output=True,
            timeout=timeout,
        )
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
    except subprocess.TimeoutExpired:
        return {"available": False, "reason": f"epubcheck timed out after {timeout}s", "findings": []}
    except (OSError, json.JSONDecodeError) as e:
        stderr = proc.stderr.decode("utf-8", "replace")[-500:] if "proc" in locals() else ""
        return {"available": False, "reason": f"epubcheck failed: {e} {stderr}", "findings": []}
    finally:
        try:
            os.unlink(report_path)
        except OSError:
            pass

    severity_map = {"FATAL": "error", "ERROR": "error", "WARNING": "warning", "INFO": "info",
                    "USAGE": "info", "SUPPRESSED": "info"}
    findings = []
    for msg in report.get("messages", []):
        level = severity_map.get(msg.get("severity", "ERROR"), "error")
        locations = msg.get("locations") or [{}]
        # One message can cover many locations; keep the first few so the
        # report stays readable while still saying where to look.
        locs = ", ".join(
            f"{loc.get('path', '?')}:{loc.get('line', '?')}" for loc in locations[:3]
        )
        if len(locations) > 3:
            locs += f" (+{len(locations) - 3} more)"
        findings.append(
            _finding(level, f"epubcheck/{msg.get('ID', '?')}", msg.get("message", ""), locs)
        )

    checker = report.get("checker", {})
    return {
        "available": True,
        "version": checker.get("checkerVersion"),
        "counts": {
            "fatal": checker.get("nFatal", 0),
            "error": checker.get("nError", 0),
            "warning": checker.get("nWarning", 0),
        },
        "findings": findings,
    }


def _parse_container(zf: zipfile.ZipFile, findings: list) -> str | None:
    """Validate META-INF/container.xml and return the OPF path it points at."""
    if "META-INF/container.xml" not in zf.namelist():
        findings.append(_finding("error", "ocf/missing-container",
                                 "META-INF/container.xml is missing"))
        return None
    try:
        tree = etree.fromstring(zf.read("META-INF/container.xml"))
    except etree.XMLSyntaxError as e:
        findings.append(_finding("error", "ocf/bad-container",
                                 f"container.xml is not well-formed XML: {e}"))
        return None
    rootfile = tree.find(f".//{{{CONTAINER_NS}}}rootfile")
    if rootfile is None or not rootfile.get("full-path"):
        findings.append(_finding("error", "ocf/no-rootfile",
                                 "container.xml declares no rootfile"))
        return None
    opf_path = rootfile.get("full-path")
    if opf_path not in zf.namelist():
        findings.append(_finding("error", "ocf/rootfile-missing",
                                 f"container.xml points at {opf_path!r}, which is not in the archive"))
        return None
    return opf_path


def _check_ocf(zf: zipfile.ZipFile, findings: list) -> None:
    """OCF container rules: mimetype entry first, uncompressed, exact content."""
    infos = zf.infolist()
    names = [i.filename for i in infos]
    if "mimetype" not in names:
        findings.append(_finding("error", "ocf/missing-mimetype",
                                 "Required 'mimetype' entry is missing"))
        return
    if names[0] != "mimetype":
        findings.append(_finding("error", "ocf/mimetype-not-first",
                                 "'mimetype' must be the first entry in the archive"))
    info = zf.getinfo("mimetype")
    if info.compress_type != zipfile.ZIP_STORED:
        findings.append(_finding("error", "ocf/mimetype-compressed",
                                 "'mimetype' must be stored uncompressed"))
    if zf.read("mimetype").rstrip() != EPUB_MIMETYPE:
        findings.append(_finding("error", "ocf/mimetype-content",
                                 "'mimetype' content must be exactly 'application/epub+zip'"))


def _iter_refs(doc: etree._Element):
    """Yield (attr, value) for every resource reference in a content document."""
    for el in doc.iter():
        if not isinstance(el.tag, str):
            continue
        tag = etree.QName(el).localname
        for attr in ("src", "href", "poster", "data"):
            v = el.get(attr)
            if v:
                yield tag, attr, v


def parse_opf(zf: zipfile.ZipFile, opf_path: str):
    """
    Parse the OPF into (manifest, spine_paths, metadata, nav_path, ncx_path).
    manifest: id -> {"href","path","media-type","properties"} with 'path'
    resolved to a full zip member name. spine_paths follow spine order.
    """
    tree = etree.fromstring(zf.read(opf_path))
    opf_dir = posixpath.dirname(opf_path)

    def resolve(href: str) -> str:
        href = unquote(urlparse(href).path)
        return posixpath.normpath(posixpath.join(opf_dir, href)) if opf_dir else posixpath.normpath(href)

    manifest = {}
    nav_path = None
    for item in tree.findall(f".//{{{OPF_NS}}}manifest/{{{OPF_NS}}}item"):
        entry = {
            "href": item.get("href", ""),
            "path": resolve(item.get("href", "")),
            "media-type": item.get("media-type", ""),
            "properties": (item.get("properties") or "").split(),
        }
        manifest[item.get("id", "")] = entry
        if "nav" in entry["properties"]:
            nav_path = entry["path"]

    spine_paths = []
    ncx_path = None
    spine = tree.find(f".//{{{OPF_NS}}}spine")
    if spine is not None:
        toc_id = spine.get("toc")
        if toc_id and toc_id in manifest:
            ncx_path = manifest[toc_id]["path"]
        for itemref in spine.findall(f"{{{OPF_NS}}}itemref"):
            idref = itemref.get("idref", "")
            if idref in manifest:
                spine_paths.append(manifest[idref]["path"])

    metadata = {}
    for tag in ("identifier", "title", "language"):
        el = tree.find(f".//{{{DC_NS}}}{tag}")
        metadata[tag] = (el.text or "").strip() if el is not None else None

    return manifest, spine_paths, metadata, nav_path, ncx_path


def _check_package(zf: zipfile.ZipFile, opf_path: str, findings: list):
    """OPF-level checks. Returns (manifest, spine_paths) for reuse downstream."""
    try:
        manifest, spine_paths, metadata, nav_path, ncx_path = parse_opf(zf, opf_path)
    except etree.XMLSyntaxError as e:
        findings.append(_finding("error", "opf/not-xml",
                                 f"OPF is not well-formed XML: {e}", opf_path))
        return {}, []

    for field, value in metadata.items():
        if not value:
            findings.append(_finding("error", "opf/missing-metadata",
                                     f"Required metadata dc:{field} is missing or empty", opf_path))

    names = set(zf.namelist())

    # Every manifest item must exist in the archive.
    for item_id, entry in manifest.items():
        if entry["path"] not in names:
            findings.append(_finding(
                "error", "opf/manifest-file-missing",
                f"Manifest item {item_id!r} points at {entry['path']!r}, not in archive", opf_path))

    # Every content file in the archive should be declared in the manifest.
    declared = {e["path"] for e in manifest.values()}
    for name in names:
        if name == "mimetype" or name.startswith("META-INF/") or name == opf_path:
            continue
        if name.endswith("/"):
            continue
        if name not in declared:
            findings.append(_finding("warning", "opf/undeclared-resource",
                                     f"{name!r} is in the archive but not declared in the manifest"))

    if not spine_paths:
        findings.append(_finding("error", "opf/empty-spine",
                                 "Spine is empty — the book has no readable content", opf_path))
    if nav_path is None and ncx_path is None:
        findings.append(_finding("error", "opf/no-navigation",
                                 "No EPUB3 nav document and no NCX — readers cannot show a TOC", opf_path))

    return manifest, spine_paths


def _check_content_documents(zf: zipfile.ZipFile, manifest: dict, findings: list) -> None:
    """
    Per-document checks on every XHTML content doc: XML well-formedness,
    XHTML namespace, and that every internal reference (img/src, link/href,
    a/href, ...) resolves to a file in the archive.
    """
    names = set(zf.namelist())
    doc_paths = sorted({
        e["path"] for e in manifest.values()
        if e["media-type"] == "application/xhtml+xml" or e["path"].endswith((".xhtml", ".html"))
    })
    for path in doc_paths:
        if path not in names:
            continue  # already reported by the manifest check
        try:
            doc = etree.fromstring(zf.read(path))
        except etree.XMLSyntaxError as e:
            findings.append(_finding("error", "content/not-xml",
                                     f"Not well-formed XML: {e}", path))
            continue

        root_ns = etree.QName(doc).namespace
        if root_ns != XHTML_NS:
            findings.append(_finding("error", "content/wrong-namespace",
                                     f"Root element namespace is {root_ns!r}, expected XHTML", path))

        base = posixpath.dirname(path)
        for tag, attr, value in _iter_refs(doc):
            parsed = urlparse(value)
            if parsed.scheme or parsed.netloc:
                if parsed.scheme in ("http", "https"):
                    findings.append(_finding(
                        "warning", "content/remote-resource",
                        f"<{tag} {attr}='{value}'> references a remote resource; "
                        "offline readers will not load it", path))
                continue
            target = unquote(parsed.path)
            if not target:
                continue  # pure fragment link
            resolved = posixpath.normpath(posixpath.join(base, target)) if base \
                else posixpath.normpath(target)
            if resolved not in names:
                findings.append(_finding(
                    "error", "content/broken-reference",
                    f"<{tag} {attr}='{value}'> does not resolve to a file in the EPUB", path))


def _check_images(zf: zipfile.ZipFile, findings: list) -> None:
    for name in zf.namelist():
        if name.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".webp")):
            try:
                img = Image.open(io.BytesIO(zf.read(name)))
                img.load()
            except (UnidentifiedImageError, OSError) as e:
                findings.append(_finding("error", "image/not-decodable",
                                         f"Image does not decode: {e}", name))


def check_format(epub_path: str, epubcheck: bool = True) -> dict:
    """
    Layer 1: structural + spec validation. Pure-Python checks always run;
    the official EPUBCheck runs on top when available.
    """
    findings: list[dict] = []
    epubcheck_result = {"available": False, "reason": "disabled", "findings": []}

    try:
        with zipfile.ZipFile(epub_path) as zf:
            bad = zf.testzip()
            if bad is not None:
                findings.append(_finding("error", "zip/corrupt",
                                         f"Archive is corrupt at member {bad!r}"))
            else:
                _check_ocf(zf, findings)
                opf_path = _parse_container(zf, findings)
                if opf_path:
                    manifest, _ = _check_package(zf, opf_path, findings)
                    _check_content_documents(zf, manifest, findings)
                _check_images(zf, findings)
    except (zipfile.BadZipFile, OSError) as e:
        findings.append(_finding("error", "zip/unreadable",
                                 f"File could not be opened as a zip archive: {e}"))
        return {"passed": False, "findings": findings, "epubcheck": epubcheck_result}

    if epubcheck:
        epubcheck_result = run_epubcheck(epub_path)
        findings.extend(epubcheck_result["findings"])

    passed = not any(f["level"] == "error" for f in findings)
    return {"passed": passed, "findings": findings, "epubcheck": epubcheck_result}


# ════════════════════════════════ Layer 2: render ════════════════════════════


def simulate_render(
    epub_path: str,
    screenshots_dir: str | None = None,
    max_screenshots: int = 5,
    max_documents: int | None = None,
) -> dict:
    """
    Layer 2: open every spine document in headless Chromium exactly as a
    WebKit/Blink-based reader would, and report what actually goes wrong at
    render time: resources that fail to load, images that decode to nothing,
    script/page errors, chapters that render blank, and content wider than
    the viewport (which forces horizontal scrolling on e-readers).

    Skipped (available=False) if Playwright or a Chromium build is missing.
    """
    try:
        from playwright.sync_api import sync_playwright, Error as PlaywrightError
    except ImportError:
        return {"available": False,
                "reason": "playwright not installed (pip install playwright)",
                "findings": [], "documents": []}

    findings: list[dict] = []
    documents: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="epub_render_") as workdir:
        try:
            with zipfile.ZipFile(epub_path) as zf:
                zf.extractall(workdir)
                opf_path = _parse_container(zf, [])
                if not opf_path:
                    return {"available": True, "findings": [
                        _finding("error", "render/no-container",
                                 "Cannot locate OPF; nothing to render")], "documents": []}
                _, spine_paths, _, nav_path, _ = parse_opf(zf, opf_path)
        except (zipfile.BadZipFile, OSError, etree.XMLSyntaxError) as e:
            return {"available": True, "findings": [
                _finding("error", "render/unpack-failed", f"Could not unpack EPUB: {e}")],
                "documents": []}

        if max_documents:
            spine_paths = spine_paths[:max_documents]

        if screenshots_dir:
            os.makedirs(screenshots_dir, exist_ok=True)

        try:
            with sync_playwright() as pw:
                try:
                    browser = pw.chromium.launch(headless=True)
                except PlaywrightError:
                    # The pip package's pinned browser revision isn't on disk;
                    # fall back to any Chromium binary we can find.
                    exe = next((p for p in CHROMIUM_FALLBACKS
                                if p and os.path.exists(p)), None)
                    if not exe:
                        raise
                    browser = pw.chromium.launch(headless=True, executable_path=exe)
                page = browser.new_page(viewport=RENDER_VIEWPORT)

                # One set of listeners for the page's lifetime; the buffers
                # are drained per document.
                console_errors: list[str] = []
                page_errors: list[str] = []
                failed: list[str] = []
                page.on("console",
                        lambda msg: console_errors.append(msg.text)
                        if msg.type == "error" else None)
                page.on("pageerror", lambda exc: page_errors.append(str(exc)))
                page.on("requestfailed",
                        lambda req: failed.append(f"{req.url} ({req.failure})"))

                for idx, doc_path in enumerate(spine_paths):
                    fs_path = os.path.join(workdir, doc_path)
                    doc_report = {
                        "document": doc_path,
                        "console_errors": [],
                        "page_errors": [],
                        "failed_requests": [],
                        "broken_images": [],
                        "visible_text_chars": 0,
                        "image_count": 0,
                        "horizontal_overflow": False,
                        "screenshot": None,
                    }
                    if not os.path.isfile(fs_path):
                        findings.append(_finding("error", "render/missing-spine-doc",
                                                 "Spine document missing from archive", doc_path))
                        documents.append(doc_report)
                        continue

                    console_errors.clear()
                    page_errors.clear()
                    failed.clear()
                    try:
                        page.goto(Path(fs_path).as_uri(), wait_until="load", timeout=30000)
                        metrics = page.evaluate(
                            """() => ({
                                text: document.body ? document.body.innerText.trim().length : 0,
                                images: document.images.length,
                                broken: Array.from(document.images)
                                    .filter(i => !i.complete || i.naturalWidth === 0)
                                    .map(i => i.getAttribute('src')),
                                overflow: (document.scrollingElement
                                    ? document.scrollingElement.scrollWidth
                                      > document.scrollingElement.clientWidth + 1
                                    : false),
                            })"""
                        )
                    except PlaywrightError as e:
                        findings.append(_finding("error", "render/load-failed",
                                                 f"Chromium could not load the document: {e}",
                                                 doc_path))
                        documents.append(doc_report)
                        continue

                    doc_report.update(
                        console_errors=list(console_errors[:10]),
                        page_errors=list(page_errors[:10]),
                        failed_requests=list(failed[:20]),
                        broken_images=metrics["broken"][:20],
                        visible_text_chars=metrics["text"],
                        image_count=metrics["images"],
                        horizontal_overflow=metrics["overflow"],
                    )

                    if metrics["broken"]:
                        findings.append(_finding(
                            "error", "render/broken-images",
                            f"{len(metrics['broken'])} image(s) fail to render: "
                            f"{', '.join(metrics['broken'][:5])}", doc_path))
                    for req in failed[:5]:
                        findings.append(_finding("error", "render/resource-failed",
                                                 f"Resource failed to load: {req}", doc_path))
                    for err in page_errors[:5]:
                        findings.append(_finding("warning", "render/page-error", err, doc_path))
                    is_nav = nav_path is not None and doc_path == nav_path
                    if metrics["text"] < 20 and metrics["images"] == 0 and not is_nav:
                        findings.append(_finding(
                            "warning", "render/blank-document",
                            f"Renders (near-)blank: {metrics['text']} visible characters, "
                            "no images", doc_path))
                    if metrics["overflow"]:
                        findings.append(_finding(
                            "warning", "render/horizontal-overflow",
                            f"Content overflows a {RENDER_VIEWPORT['width']}px viewport "
                            "horizontally; e-readers will clip or force panning", doc_path))

                    if screenshots_dir and idx < max_screenshots:
                        shot = os.path.join(
                            screenshots_dir,
                            f"{idx:03d}_{Path(doc_path).stem}.png")
                        try:
                            page.screenshot(path=shot, full_page=False)
                            doc_report["screenshot"] = shot
                        except PlaywrightError:
                            pass

                    documents.append(doc_report)

                browser.close()
        except PlaywrightError as e:
            return {"available": False,
                    "reason": f"Chromium could not be launched: {e}",
                    "findings": findings, "documents": documents}

    total_text = sum(d["visible_text_chars"] for d in documents)
    if documents and total_text < 200:
        findings.append(_finding(
            "error", "render/book-blank",
            f"The whole book renders only {total_text} visible characters"))

    passed = not any(f["level"] == "error" for f in findings)
    return {"available": True, "passed": passed, "findings": findings,
            "documents": documents,
            "totals": {
                "documents_rendered": len(documents),
                "visible_text_chars": total_text,
                "broken_images": sum(len(d["broken_images"]) for d in documents),
            }}


# ═══════════════════════════════ Layer 3: fidelity ═══════════════════════════

_WS_RE = re.compile(r"\s+")
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\s*\n\s*(\w)")
# UTF-8 bytes mis-decoded as Latin-1/cp1252 leave 'Â'/'â' followed by
# punctuation-range garbage; a handful of legitimate words do contain 'â',
# so require the tell-tale trailing byte.
_MOJIBAKE_RE = re.compile(r"[ÂÃ][-¿–—‘’“”®©]")


def normalize_text(text: str) -> str:
    """Normalize text so PDF extraction and EPUB extraction become comparable."""
    text = unicodedata.normalize("NFKC", text)  # folds ligatures (ﬁ → fi) etc.
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)  # re-join line-break hyphenation
    text = text.replace("­", "")           # soft hyphens
    text = _WS_RE.sub(" ", text)
    return text.strip().casefold()


def _shingles(words: list[str], n: int = 5) -> set:
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def extract_epub_text(epub_path: str) -> list[dict]:
    """Extract visible text per spine document, in spine order."""
    out = []
    with zipfile.ZipFile(epub_path) as zf:
        opf_path = _parse_container(zf, [])
        if not opf_path:
            return out
        _, spine_paths, _, nav_path, _ = parse_opf(zf, opf_path)
        names = set(zf.namelist())
        for path in spine_paths:
            if path == nav_path or path not in names:
                continue
            try:
                doc = etree.fromstring(zf.read(path))
            except etree.XMLSyntaxError:
                # Fall back to a lenient parse so fidelity can still be
                # measured on a book whose format layer already failed.
                doc = etree.fromstring(zf.read(path),
                                       parser=etree.HTMLParser(recover=True))
                if doc is None:
                    continue
            for el in doc.iter():
                if isinstance(el.tag, str) and etree.QName(el).localname in ("script", "style"):
                    el.clear()
            text = " ".join(t for t in doc.itertext())
            out.append({"document": path, "text": text})
    return out


def extract_pdf_pages(pdf_path: str) -> tuple[list[str], list, int]:
    """Return (per-page text, toc entries, distinct raster image count)."""
    import fitz

    doc = fitz.open(pdf_path)
    try:
        pages = [page.get_text() for page in doc]
        toc = doc.get_toc(simple=True)
        xrefs = set()
        for page in doc:
            for img in page.get_images(full=True):
                xrefs.add(img[0])
        return pages, toc, len(xrefs)
    finally:
        doc.close()


def compare_with_pdf(epub_path: str, pdf_path: str, shingle_size: int = 5) -> dict:
    """
    Layer 3: how faithful is the EPUB to the source PDF?

    Coverage is measured with word n-gram (shingle) containment, which is
    robust to OCR line-wrapping and hyphenation differences and stays O(n)
    on book-length texts:

      * per-PDF-page coverage — the fraction of each page's shingles found
        anywhere in the EPUB; pages below 50% are reported as missing spans.
      * extra ratio — the fraction of EPUB shingles absent from the PDF
        (OCR hallucination / boilerplate injection).
      * TOC coverage — how many PDF bookmark titles appear in the EPUB text.
    """
    findings: list[dict] = []

    pdf_pages_raw, pdf_toc, pdf_image_count = extract_pdf_pages(pdf_path)
    epub_docs = extract_epub_text(epub_path)

    pdf_pages = [normalize_text(p) for p in pdf_pages_raw]
    pdf_full = " ".join(pdf_pages)
    epub_full = normalize_text(" ".join(d["text"] for d in epub_docs))

    pdf_words = pdf_full.split()
    epub_words = epub_full.split()

    epub_shingles = _shingles(epub_words, shingle_size)
    pdf_shingles = _shingles(pdf_words, shingle_size)

    # Per-page coverage → overall coverage + location of the gaps.
    page_coverage: list[float] = []
    for page_text in pdf_pages:
        words = page_text.split()
        page_sh = _shingles(words, shingle_size)
        if not page_sh:
            page_coverage.append(1.0)  # blank page: nothing to lose
            continue
        found = sum(1 for s in page_sh if s in epub_shingles)
        page_coverage.append(found / len(page_sh))

    content_pages = [c for c, p in zip(page_coverage, pdf_pages) if p.split()]
    text_coverage = (sum(content_pages) / len(content_pages)) if content_pages else 0.0

    # Contiguous runs of poorly-covered pages → human-readable missing spans.
    missing_spans = []
    run_start = None
    for i, cov in enumerate(page_coverage):
        poor = cov < 0.5 and bool(pdf_pages[i].split())
        if poor and run_start is None:
            run_start = i
        elif not poor and run_start is not None:
            missing_spans.append((run_start + 1, i))  # 1-based, inclusive
            run_start = None
    if run_start is not None:
        missing_spans.append((run_start + 1, len(page_coverage)))

    extra_ratio = 0.0
    if epub_shingles:
        extra = sum(1 for s in epub_shingles if s not in pdf_shingles)
        extra_ratio = extra / len(epub_shingles)

    # TOC coverage: PDF bookmarks that survived into the EPUB text.
    toc_total = toc_found = 0
    toc_missing = []
    for _level, title, _page in pdf_toc:
        norm = normalize_text(title)
        if len(norm) < 4:
            continue
        toc_total += 1
        if norm in epub_full:
            toc_found += 1
        else:
            toc_missing.append(title)
    toc_coverage = (toc_found / toc_total) if toc_total else None

    # EPUB-side counts.
    epub_image_count = 0
    try:
        with zipfile.ZipFile(epub_path) as zf:
            epub_image_count = sum(
                1 for n in zf.namelist()
                if n.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".webp"))
                and "cover" not in n.lower())
    except (zipfile.BadZipFile, OSError):
        pass

    mojibake = len(_MOJIBAKE_RE.findall(epub_full))

    # ── findings from the metrics ──
    if text_coverage < COVERAGE_ERROR_BELOW:
        findings.append(_finding(
            "error", "fidelity/low-text-coverage",
            f"Only {text_coverage:.0%} of the PDF's text is present in the EPUB "
            f"(threshold {COVERAGE_ERROR_BELOW:.0%})"))
    elif text_coverage < COVERAGE_WARN_BELOW:
        findings.append(_finding(
            "warning", "fidelity/reduced-text-coverage",
            f"{text_coverage:.0%} of the PDF's text is present in the EPUB "
            f"(threshold {COVERAGE_WARN_BELOW:.0%})"))
    for start, end in missing_spans[:20]:
        findings.append(_finding(
            "warning", "fidelity/missing-span",
            f"PDF pages {start}–{end} are largely absent from the EPUB"
            if end > start else f"PDF page {start} is largely absent from the EPUB"))
    if len(missing_spans) > 20:
        findings.append(_finding(
            "warning", "fidelity/missing-span",
            f"...and {len(missing_spans) - 20} more poorly-covered page ranges"))
    if extra_ratio > 0.20:
        findings.append(_finding(
            "warning", "fidelity/extra-content",
            f"{extra_ratio:.0%} of the EPUB's text has no counterpart in the PDF "
            "(possible OCR hallucination or repeated boilerplate)"))
    if toc_coverage is not None and toc_coverage < 0.75:
        findings.append(_finding(
            "warning", "fidelity/toc-entries-missing",
            f"Only {toc_found}/{toc_total} PDF bookmark titles found in the EPUB text; "
            f"missing e.g.: {', '.join(toc_missing[:5])}"))
    if pdf_image_count > 0 and epub_image_count == 0:
        findings.append(_finding(
            "warning", "fidelity/images-dropped",
            f"PDF contains {pdf_image_count} images; the EPUB contains none "
            "(figures/diagrams were lost in conversion)"))
    if mojibake:
        findings.append(_finding(
            "warning", "fidelity/mojibake",
            f"{mojibake} probable encoding-corruption sequences (e.g. 'Â®') in the EPUB text"))

    chapter_count = len(epub_docs)
    if len(pdf_pages) >= 50 and chapter_count <= 1:
        findings.append(_finding(
            "warning", "fidelity/no-chapter-structure",
            f"A {len(pdf_pages)}-page PDF became a single spine document; "
            "chapter splitting failed"))

    passed = not any(f["level"] == "error" for f in findings)
    return {
        "passed": passed,
        "findings": findings,
        "metrics": {
            "pdf_pages": len(pdf_pages),
            "pdf_words": len(pdf_words),
            "epub_words": len(epub_words),
            "word_ratio": round(len(epub_words) / len(pdf_words), 3) if pdf_words else None,
            "text_coverage": round(text_coverage, 3),
            "extra_ratio": round(extra_ratio, 3),
            "toc_coverage": round(toc_coverage, 3) if toc_coverage is not None else None,
            "toc_missing_titles": toc_missing[:25],
            "missing_page_spans": missing_spans[:50],
            "pdf_image_count": pdf_image_count,
            "epub_image_count": epub_image_count,
            "epub_spine_documents": chapter_count,
            "mojibake_sequences": mojibake,
        },
    }


# ═══════════════════════════════════ Aggregate ═══════════════════════════════


def validate_full(
    epub_path: str,
    pdf_path: str | None = None,
    render: bool = True,
    epubcheck: bool = True,
    screenshots_dir: str | None = None,
) -> dict:
    """
    Run every applicable layer and aggregate into a single report with an
    overall verdict: "fail" if any layer produced an error, "warn" if only
    warnings, "pass" otherwise. Layers whose tooling is unavailable are
    marked skipped and do not affect the verdict.
    """
    report: dict = {"epub": epub_path, "pdf": pdf_path}

    report["format"] = check_format(epub_path, epubcheck=epubcheck)

    if render:
        report["render"] = simulate_render(epub_path, screenshots_dir=screenshots_dir)
    else:
        report["render"] = {"available": False, "reason": "disabled", "findings": []}

    if pdf_path:
        try:
            report["fidelity"] = compare_with_pdf(epub_path, pdf_path)
        except Exception as e:  # a broken PDF must not mask the other layers
            logger.exception("PDF comparison failed")
            report["fidelity"] = {
                "passed": False,
                "findings": [_finding("error", "fidelity/comparison-failed",
                                      f"Could not compare against PDF: {e}")],
                "metrics": {},
            }
    else:
        report["fidelity"] = None

    all_findings = list(report["format"]["findings"])
    all_findings += report["render"].get("findings", [])
    if report["fidelity"]:
        all_findings += report["fidelity"]["findings"]

    errors = [f for f in all_findings if f["level"] == "error"]
    warnings_ = [f for f in all_findings if f["level"] == "warning"]
    report["verdict"] = "fail" if errors else ("warn" if warnings_ else "pass")
    report["counts"] = {"errors": len(errors), "warnings": len(warnings_)}

    summary = []
    fmt = report["format"]
    ec = fmt["epubcheck"]
    summary.append(
        f"Format: {'PASS' if fmt['passed'] else 'FAIL'}"
        + (f" (EPUBCheck {ec.get('version')}: {ec['counts']['fatal']} fatal, "
           f"{ec['counts']['error']} errors, {ec['counts']['warning']} warnings)"
           if ec.get("available") else " (EPUBCheck unavailable)"))
    rnd = report["render"]
    if rnd.get("available"):
        t = rnd["totals"]
        summary.append(
            f"Render: {'PASS' if rnd['passed'] else 'FAIL'} — "
            f"{t['documents_rendered']} documents, {t['visible_text_chars']} visible chars, "
            f"{t['broken_images']} broken images")
    else:
        summary.append(f"Render: SKIPPED ({rnd.get('reason')})")
    if report["fidelity"]:
        m = report["fidelity"]["metrics"]
        if m:
            summary.append(
                f"Fidelity vs PDF: {'PASS' if report['fidelity']['passed'] else 'FAIL'} — "
                f"text coverage {m['text_coverage']:.0%}"
                + (f", TOC coverage {m['toc_coverage']:.0%}"
                   if m.get("toc_coverage") is not None else "")
                + f", images {m['epub_image_count']}/{m['pdf_image_count']}")
        else:
            summary.append("Fidelity vs PDF: FAIL — comparison could not run")
    else:
        summary.append("Fidelity: skipped (no source PDF supplied)")
    report["summary"] = summary

    return report


# ═══════════════════════════════════ CLI ═════════════════════════════════════


def _main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Full EPUB validation: format, simulated render, PDF fidelity")
    parser.add_argument("epub", help="EPUB file to validate")
    parser.add_argument("pdf", nargs="?", help="Source PDF to compare against")
    parser.add_argument("--no-render", action="store_true", help="Skip render simulation")
    parser.add_argument("--no-epubcheck", action="store_true", help="Skip EPUBCheck")
    parser.add_argument("--screenshots", metavar="DIR", help="Save render screenshots here")
    parser.add_argument("--json", metavar="FILE", help="Write the full JSON report here")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    report = validate_full(
        args.epub,
        pdf_path=args.pdf,
        render=not args.no_render,
        epubcheck=not args.no_epubcheck,
        screenshots_dir=args.screenshots,
    )

    print(f"\n════ {os.path.basename(args.epub)} ════")
    for line in report["summary"]:
        print(" ", line)
    print(f"\nVerdict: {report['verdict'].upper()} "
          f"({report['counts']['errors']} errors, {report['counts']['warnings']} warnings)\n")

    shown = 0
    for section in ("format", "render", "fidelity"):
        block = report.get(section)
        if not block:
            continue
        for f in block.get("findings", []):
            loc = f" [{f['location']}]" if f.get("location") else ""
            print(f"  {f['level'].upper():7s} {f['code']}: {f['message']}{loc}")
            shown += 1
            if shown >= 60:
                print("  ... (further findings elided; use --json for the full report)")
                break
        if shown >= 60:
            break

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fp:
            json.dump(report, fp, indent=2, ensure_ascii=False)
        print(f"\nFull report written to {args.json}")

    return 0 if report["verdict"] == "pass" else 1


if __name__ == "__main__":
    import sys

    sys.exit(_main(sys.argv[1:]))
