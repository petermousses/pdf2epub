"""
Build an EPUB from OCR-extracted markdown text and an optional cover image.
"""

import html
import io
import os
import posixpath
import re
import shutil
import uuid
import logging
import warnings
import zipfile
from pathlib import Path

import markdown
import lxml.html as lhtml
from lxml import etree
from lxml.html import html5parser
from ebooklib import epub
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

XHTML_NS = "http://www.w3.org/1999/xhtml"
XML_NS = "http://www.w3.org/XML/1998/namespace"
OPS_NS = "http://www.idpf.org/2007/ops"

# A single oversized XHTML file (a whole book in one spine document) makes
# e-readers slow to paginate and some crash outright; chapters larger than
# this are split at paragraph boundaries.
MAX_CHAPTER_CHARS = int(os.environ.get("EPUB_MAX_CHAPTER_CHARS", "30000"))

# ── Content whitelist ────────────────────────────────────────────────────────
# OCR output routinely contains stray angle-bracket sequences that markdown
# passes through as raw HTML. Some of those parse into *well-formed* XML that
# is nevertheless invalid EPUB content (<page>, <name>, <xsl:template>, ...),
# which strict readers and EPUBCheck reject. Everything the builder ships is
# therefore filtered against the XHTML elements markdown can legitimately
# produce; anything else is unwrapped (tags dropped, text kept).
ALLOWED_TAGS = {
    "html", "head", "title", "meta", "link", "style", "body",
    "h1", "h2", "h3", "h4", "h5", "h6", "p", "div", "span", "section",
    "nav", "article", "aside", "header", "footer", "figure", "figcaption",
    "a", "em", "strong", "b", "i", "u", "s", "small", "sub", "sup", "br",
    "hr", "abbr", "cite", "q", "dfn", "kbd", "samp", "var", "time", "mark",
    "ins", "del", "ul", "ol", "li", "dl", "dt", "dd", "blockquote", "pre",
    "code", "table", "caption", "thead", "tbody", "tfoot", "tr", "th", "td",
    "img", "details", "summary",
}

# Elements that are only valid inside specific parents; markdown never emits
# them elsewhere, so a violation is always OCR garbage.
PARENT_REQUIRED = {
    "li": {"ul", "ol"},
    "dt": {"dl"}, "dd": {"dl"},
    "tr": {"table", "thead", "tbody", "tfoot"},
    "td": {"tr"}, "th": {"tr"},
    "thead": {"table"}, "tbody": {"table"}, "tfoot": {"table"},
    "caption": {"table"},
}

GLOBAL_ATTRS = {"id", "class", "title", "lang", "dir", "style", "role"}
TAG_ATTRS = {
    "a": {"href", "rel"},
    "img": {"src", "alt", "width", "height"},
    "link": {"href", "rel", "type", "media"},
    "meta": {"name", "content", "charset", "http-equiv"},
    "th": {"colspan", "rowspan", "scope"},
    "td": {"colspan", "rowspan"},
    "ol": {"start", "type"},
    "time": {"datetime"},
}
# Namespaced attributes that legitimately appear in our documents (nav docs
# carry epub:type; ebooklib's wrapper sets xml:lang and epub:prefix).
ALLOWED_NS_ATTRS = {
    f"{{{XML_NS}}}lang", f"{{{XML_NS}}}space",
    f"{{{OPS_NS}}}type", f"{{{OPS_NS}}}prefix",
}

IMG_REF_RE = re.compile(r"images/page_(\d+)_(\d+)\.(jpe?g|png|gif|webp)", re.I)


def _split_chapters(text: str) -> list[tuple[str, str]]:
    """
    Split markdown text into (title, content) chapters at H1/H2 headings
    (up to 3 leading spaces tolerated, as in the markdown spec).
    Falls back to a single chapter if no headings found.
    """
    chapter_re = re.compile(r"^\s{0,3}(#{1,2})\s+(.+)$", re.MULTILINE)
    matches = list(chapter_re.finditer(text))

    if not matches:
        return [("Document", text)]

    chapters = []
    for i, match in enumerate(matches):
        title = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        chapters.append((title, content))

    return chapters


def _split_oversized(
    chapters: list[tuple[str, str]], max_chars: int = MAX_CHAPTER_CHARS
) -> list[tuple[str, str]]:
    """
    Break any chapter larger than max_chars into parts at paragraph
    boundaries. OCR output for a whole book sometimes contains no usable
    headings at all, and a 600-page book shipped as one XHTML file
    paginates painfully slowly (or crashes) on real readers.
    """
    out = []
    for title, content in chapters:
        if len(content) <= max_chars:
            out.append((title, content))
            continue
        paragraphs = re.split(r"\n\s*\n", content)
        parts: list[str] = []
        current: list[str] = []
        current_len = 0
        for para in paragraphs:
            if current and current_len + len(para) > max_chars:
                parts.append("\n\n".join(current))
                current, current_len = [], 0
            current.append(para)
            current_len += len(para) + 2
        if current:
            parts.append("\n\n".join(current))
        if len(parts) <= 1:
            out.append((title, content))
        else:
            logger.info("Chapter %r (%d chars) split into %d parts",
                        title, len(content), len(parts))
            out.extend(
                (f"{title} ({i}/{len(parts)})", part)
                for i, part in enumerate(parts, 1)
            )
    return out


def extract_pdf_images_for_markdown(pdf_path: str, ocr_text: str) -> dict[str, bytes]:
    """
    Resolve image references the OCR model emits in its markdown
    ('images/page_<page>_<index>.jpg') to actual image bytes pulled out of
    the source PDF, so the EPUB can package them instead of shipping broken
    <img> tags. Unresolvable references return no entry — the sanitizer
    then strips those tags.

    Images are re-encoded to match the referenced extension (readers and
    EPUBCheck expect content, manifest media-type, and extension to agree).
    """
    refs: dict[tuple[int, int], str] = {}
    for m in IMG_REF_RE.finditer(ocr_text):
        refs.setdefault((int(m.group(1)), int(m.group(2))), m.group(0))
    if not refs:
        return {}

    import fitz  # deferred: only conversion jobs need PyMuPDF here

    out: dict[str, bytes] = {}
    doc = fitz.open(pdf_path)
    try:
        for (page_no, img_no), ref in sorted(refs.items()):
            if not 0 <= page_no < len(doc):
                logger.warning("OCR referenced %s but PDF has no page %d", ref, page_no)
                continue
            images = doc[page_no].get_images(full=True)
            if not 0 <= img_no < len(images):
                logger.warning("OCR referenced %s but page %d has %d image(s)",
                               ref, page_no, len(images))
                continue
            xref = images[img_no][0]
            try:
                raw = doc.extract_image(xref)["image"]
                img = Image.open(io.BytesIO(raw))
                img.load()
                target_jpeg = ref.lower().endswith((".jpg", ".jpeg"))
                buf = io.BytesIO()
                if target_jpeg:
                    if img.mode not in ("RGB", "L"):
                        img = img.convert("RGB")
                    img.save(buf, format="JPEG", quality=90)
                else:
                    if img.mode == "CMYK":
                        img = img.convert("RGB")
                    img.save(buf, format="PNG")
                out[ref] = buf.getvalue()
            except (UnidentifiedImageError, OSError, ValueError) as e:
                logger.warning("Could not extract %s from PDF: %s", ref, e)
    finally:
        doc.close()

    logger.info("Extracted %d/%d OCR-referenced images from the PDF",
                len(out), len(refs))
    return out


def _localname(el) -> str | None:
    """Lowercased tag name without namespace, or None for comments/PIs."""
    tag = el.tag
    if not isinstance(tag, str):
        return None
    if "}" in tag:
        tag = tag.rsplit("}", 1)[1]
    return tag.lower()


def _remove_element(el) -> None:
    """Delete an element but keep its tail text in place."""
    parent = el.getparent()
    if el.tail:
        prev = el.getprevious()
        if prev is not None:
            prev.tail = (prev.tail or "") + el.tail
        else:
            parent.text = (parent.text or "") + el.tail
    parent.remove(el)


def _unwrap_element(el) -> None:
    """Replace an element with its own text and children (drop just the tags)."""
    parent = el.getparent()
    pos = parent.index(el)
    children = list(el)
    if el.text:
        if pos > 0:
            sib = parent[pos - 1]
            sib.tail = (sib.tail or "") + el.text
        else:
            parent.text = (parent.text or "") + el.text
    for i, child in enumerate(children):
        parent.insert(pos + i, child)
    if el.tail:
        if children:
            children[-1].tail = (children[-1].tail or "") + el.tail
        elif pos > 0:
            parent[pos - 1].tail = (parent[pos - 1].tail or "") + el.tail
        else:
            parent.text = (parent.text or "") + el.tail
    parent.remove(el)


def _clean_attributes(el, local: str) -> int:
    allowed = GLOBAL_ATTRS | TAG_ATTRS.get(local, set())
    dropped = 0
    for name in list(el.attrib):
        if name in ALLOWED_NS_ATTRS:
            continue
        if name.startswith("aria-") or name.startswith("data-"):
            continue
        if name in allowed:
            continue
        del el.attrib[name]
        dropped += 1
    return dropped


def _sanitize_tree(root, allowed_image_srcs: frozenset | None = None) -> int:
    """
    Enforce the content whitelist on a parsed (X)HTML tree, in place.

    * comments/PIs are removed;
    * elements outside ALLOWED_TAGS (or violating PARENT_REQUIRED nesting)
      are unwrapped — their tags vanish, their text/children stay;
    * attributes outside the allowlist are dropped (this is what scrubs the
      'xmlnsU0003Aepub'-style names html5lib coercion leaves behind);
    * <img> whose src is not in allowed_image_srcs is removed, so no
      document ever ships a reference to an image that isn't packaged.

    Returns the number of modifications made (0 = tree was already clean).
    """
    changes = 0
    elements = list(root.iter())
    # Reverse document order processes children before their parents, so an
    # unwrapped element's children have already been vetted.
    for el in reversed(elements):
        local = _localname(el)
        if el is root:
            if local is not None:
                changes += _clean_attributes(el, local)
            continue
        if local is None:  # comment / processing instruction
            _remove_element(el)
            changes += 1
            continue
        if local not in ALLOWED_TAGS or ":" in local:
            _unwrap_element(el)
            changes += 1
            continue
        required = PARENT_REQUIRED.get(local)
        if required is not None:
            parent_local = _localname(el.getparent()) or ""
            if parent_local not in required:
                _unwrap_element(el)
                changes += 1
                continue
        if local == "img":
            src = el.get("src")
            if not src or (allowed_image_srcs is not None
                           and src not in allowed_image_srcs):
                _remove_element(el)
                changes += 1
                continue
        changes += _clean_attributes(el, local)
    return changes


def _sanitize_html_fragment(
    body: str,
    chapter_title: str,
    allowed_image_srcs: frozenset = frozenset(),
) -> str:
    """
    Turn python-markdown output into guaranteed-valid XHTML body content.

    OCR'd source text routinely contains stray '<'/'>' sequences (generics
    like "List<Item>", comparisons, page markers like "<page>") that
    python-markdown passes through as raw HTML. Two failure modes result:
    output that isn't well-formed XML at all, and output that IS well-formed
    but uses elements/attributes that are not valid EPUB content — strict
    e-reader firmware and EPUBCheck reject both. Every fragment is therefore
    parsed leniently and filtered against the content whitelist before it
    ships; if even that fails, the fragment degrades to escaped literal text
    rather than a broken chapter.
    """
    try:
        fragment = lhtml.fragment_fromstring(body, create_parent="div")
        changes = _sanitize_tree(fragment, allowed_image_srcs)
        parts = [html.escape(fragment.text) if fragment.text else ""]
        parts += [etree.tostring(child, encoding="unicode", method="xml")
                  for child in fragment]
        out = "".join(parts)
        # Belt and braces: the result must be embeddable in an XHTML doc.
        etree.fromstring(f"<div xmlns='{XHTML_NS}'>{out}</div>".encode("utf-8"))
        if changes:
            logger.warning(
                "Chapter %r: sanitized %d invalid HTML construct(s) from OCR output",
                chapter_title, changes,
            )
        return out
    except (etree.XMLSyntaxError, etree.ParserError, ValueError) as e:
        logger.warning(
            "Chapter %r: could not sanitize markdown output (%s); "
            "falling back to escaped text", chapter_title, e,
        )
        return f"<p>{html.escape(body)}</p>"


def _fix_text_encoding(text: str) -> str:
    """
    Repair mojibake in OCR output (e.g. 'O'REILLYÂ®' for 'O'REILLY®' —
    UTF-8 bytes decoded as Latin-1 somewhere upstream) before it is baked
    into the book. ftfy detects and undoes these double-encodings without
    touching already-correct text.
    """
    try:
        import ftfy
    except ImportError:
        logger.warning("ftfy not installed; skipping mojibake repair")
        return text
    fixed = ftfy.fix_text(text)
    if fixed != text:
        logger.info("Repaired encoding damage in OCR text")
    return fixed


def _prepare_cover(image_bytes: bytes) -> tuple[bytes, str, str] | None:
    """
    Validate and normalize a cover image for maximum e-reader compatibility.

    Returns (jpeg_bytes, filename, mime), or None if the image is unusable
    (in which case the EPUB is built without a cover rather than shipping a
    file that fails on the reader).
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()  # force full decode now, not lazily when a reader opens it
    except (UnidentifiedImageError, OSError, ValueError) as e:
        logger.warning("Cover image failed validation, skipping cover: %s", e)
        return None

    if img.width < 1 or img.height < 1:
        logger.warning(
            "Cover image has invalid dimensions %sx%s, skipping cover",
            img.width,
            img.height,
        )
        return None

    # Flatten to plain RGB and re-encode as baseline JPEG, the cover format
    # with the broadest e-reader support. Source renders are PNG and may
    # carry an alpha channel or a palette mode; some e-reader firmware
    # rejects those (or CMYK JPEGs) outright instead of falling back.
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue(), "cover.jpg", "image/jpeg"


def _collect_issues(path: str) -> list[str]:
    """
    Hard structural problems in an EPUB: corrupt archive, missing mimetype,
    content documents that aren't well-formed XML, or images that don't
    decode. Shared by the build-time validator and the library validate/fix
    endpoints. Raises zipfile.BadZipFile/OSError if the file can't be opened
    as a zip at all — callers decide how to report that.
    """
    issues = []
    with zipfile.ZipFile(path) as zf:
        bad = zf.testzip()
        if bad is not None:
            issues.append(f"Archive is corrupt at member: {bad}")
            return issues

        names = zf.namelist()
        if "mimetype" not in names:
            issues.append("Missing required 'mimetype' entry")

        for name in names:
            if name.endswith((".xhtml", ".html", ".ncx", ".opf")):
                try:
                    etree.fromstring(zf.read(name))
                except etree.XMLSyntaxError as e:
                    issues.append(f"{name}: not well-formed XML ({e})")
            elif name.endswith((".jpg", ".jpeg", ".png")):
                try:
                    img = Image.open(io.BytesIO(zf.read(name)))
                    img.load()
                except (UnidentifiedImageError, OSError) as e:
                    issues.append(f"{name}: image is not decodable ({e})")
    return issues


def _validate_epub(output_path: str) -> None:
    """
    Sanity-check the EPUB actually written to disk before declaring success,
    rather than letting a corrupt file reach the user's reader silently.
    """
    try:
        issues = _collect_issues(output_path)
    except (zipfile.BadZipFile, OSError) as e:
        raise RuntimeError(f"EPUB archive could not be opened: {e}") from e
    if issues:
        raise RuntimeError("EPUB validation failed: " + "; ".join(issues))


def _find_cover_href(opf_bytes: bytes) -> str | None:
    """Locate the cover image's manifest href from an EPUB's OPF (epub3 or epub2 style)."""
    tree = etree.fromstring(opf_bytes)
    ns = {"opf": "http://www.idpf.org/2007/opf"}

    for item in tree.findall(".//opf:manifest/opf:item", ns):
        if "cover-image" in (item.get("properties") or "").split():
            return item.get("href")

    cover_id = None
    for meta in tree.findall(".//opf:metadata/opf:meta", ns):
        if meta.get("name") == "cover":
            cover_id = meta.get("content")
            break
    if cover_id:
        for item in tree.findall(".//opf:manifest/opf:item", ns):
            if item.get("id") == cover_id:
                return item.get("href")
    return None


def _resolve_href(opf_name: str, href: str) -> str:
    """Resolve a manifest href (relative to the OPF's own directory) to a full zip member path."""
    if "/" not in opf_name:
        return href
    return opf_name.rsplit("/", 1)[0] + "/" + href


def _reparse_as_xhtml(data: bytes) -> bytes:
    """
    Lenient reparse of a broken (X)HTML content document into well-formed
    XML, for repair_epub.

    Uses the HTML5 parsing algorithm (via html5lib) rather than libxml2's
    HTML recovery mode: real-world OCR garbage can contain characters that
    are invalid even in a tag/attribute *name* (e.g. a literal "<" inside
    one, from mis-rendered markdown), and libxml2's recovery mode carries
    those straight through into a tree that etree.tostring then can't
    serialize as XML at all. html5lib instead coerces invalid names into
    XML-safe ones per spec, so it reliably produces well-formed output.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # DataLossWarning on name coercion
        doc = html5parser.fromstring(data)
    return etree.tostring(doc, encoding="utf-8", method="xml")


def _repair_cover_manifest_entry(
    opf_bytes: bytes, old_href: str, new_href: str, new_mime: str
) -> bytes:
    """Point the manifest <item> for the cover at its re-encoded replacement."""
    tree = etree.fromstring(opf_bytes)
    ns = {"opf": "http://www.idpf.org/2007/opf"}
    for item in tree.findall(".//opf:manifest/opf:item", ns):
        if item.get("href") == old_href:
            item.set("href", new_href)
            item.set("media-type", new_mime)
    return etree.tostring(tree, xml_declaration=True, encoding="utf-8")


def validate_epub_report(path: str) -> list[str]:
    """
    Non-raising validation for an already-built EPUB sitting in the output
    library (as opposed to _validate_epub, which raises during build). Adds
    one soft hint on top of the hard structural checks: flags a cover image
    that isn't already a flattened baseline JPEG, since that's the format
    _prepare_cover normalizes to for new builds and some e-reader firmware
    rejects other covers outright even though they decode fine.
    """
    try:
        issues = _collect_issues(path)
    except (zipfile.BadZipFile, OSError) as e:
        return [f"Could not open as a zip archive: {e}"]

    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            opf_name = next((n for n in names if n.endswith(".opf")), None)
            if opf_name:
                cover_href = _find_cover_href(zf.read(opf_name))
                cover_path = _resolve_href(opf_name, cover_href) if cover_href else None
                if cover_path and cover_path in names:
                    img = Image.open(io.BytesIO(zf.read(cover_path)))
                    img.load()
                    if img.format != "JPEG" or img.mode != "RGB":
                        issues.append(
                            f"{cover_path}: cover is {img.format}/{img.mode}, not a "
                            "flattened baseline JPEG — some e-readers reject this"
                        )
    except (zipfile.BadZipFile, OSError, etree.XMLSyntaxError, UnidentifiedImageError):
        pass  # best-effort hint on top of the hard checks already collected

    return issues


def repair_epub(path: str) -> dict:
    """
    Attempt to repair an EPUB previously written by this app in place:
    re-serialize any content document that isn't well-formed XML, scrub all
    content documents against the same element/attribute whitelist new
    builds use (removing OCR-passthrough garbage and broken image
    references), and re-encode the cover image as a flattened baseline JPEG
    if it isn't one already (see validate_epub_report). Assumes this app's own package
    layout — one flat directory holding the OPF, all content documents, and
    images, with manifest hrefs as bare filenames relative to it — so
    reference rewrites are plain basename substitutions once resolved
    against the OPF's directory.

    Anything else wrong with the file (corrupt archive, missing mimetype,
    non-cover images that fail to decode) isn't safely auto-fixable and is
    reported instead; those files need to be reconverted from the source PDF.

    The original file is preserved at "<path>.bak" before being overwritten.
    """
    issues_before = validate_epub_report(path)

    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            contents = {name: zf.read(name) for name in names}
    except (zipfile.BadZipFile, OSError) as e:
        return {
            "changed": False,
            "issues_before": issues_before,
            "issues_after": issues_before,
            "unfixable": [f"Archive could not be opened: {e}"],
        }

    changed = False
    unfixable = []

    # 1. Repair XHTML/HTML content documents: reparse the ones that aren't
    # well-formed XML, then run every document through the content-whitelist
    # sanitizer. The sanitizer removes what strict readers/EPUBCheck reject
    # even in well-formed files: bogus elements from OCR passthrough
    # (<page>, <name>, ...), mangled attribute names left by an earlier
    # html5lib repair ('xmlnsU0003Aepub'), and <img> references to files
    # that aren't in the archive. Docs are only rewritten when something
    # actually changed.
    archive_names = [n for n in contents
                     if n != "mimetype" and not n.startswith("META-INF/")]
    for name in list(contents):
        if not name.endswith((".xhtml", ".html")):
            continue
        data = contents[name]
        was_malformed = False
        try:
            doc = etree.fromstring(data)
        except etree.XMLSyntaxError:
            was_malformed = True
            try:
                repaired = _reparse_as_xhtml(data)
                doc = etree.fromstring(repaired)  # verify the repair actually took
            except Exception as e:
                unfixable.append(f"{name}: could not be repaired ({e})")
                continue

        base = posixpath.dirname(name)
        allowed_srcs = frozenset(
            posixpath.relpath(member, base) if base else member
            for member in archive_names
        )
        sanitize_changes = _sanitize_tree(doc, allowed_srcs)
        if not (was_malformed or sanitize_changes):
            continue
        contents[name] = etree.tostring(
            doc, xml_declaration=True, encoding="utf-8")
        changed = True

    # 2. Re-normalize the cover image if it isn't already a baseline JPEG.
    opf_name = next((n for n in names if n.endswith(".opf")), None)
    if opf_name:
        try:
            cover_href = _find_cover_href(contents[opf_name])
        except etree.XMLSyntaxError as e:
            cover_href = None
            unfixable.append(f"{opf_name}: could not read manifest ({e})")

        cover_path = _resolve_href(opf_name, cover_href) if cover_href else None
        if cover_path and cover_path in contents:
            cover_bytes = contents[cover_path]
            is_baseline_jpeg = False
            try:
                img = Image.open(io.BytesIO(cover_bytes))
                img.load()
                is_baseline_jpeg = img.format == "JPEG" and img.mode == "RGB"
            except (UnidentifiedImageError, OSError) as e:
                unfixable.append(f"{cover_path}: cover image is not decodable ({e})")

            if not is_baseline_jpeg:
                prepared = _prepare_cover(cover_bytes)
                if prepared is None:
                    unfixable.append(f"{cover_path}: cover image could not be re-encoded")
                else:
                    new_bytes, new_href, new_mime = prepared
                    new_path = _resolve_href(opf_name, new_href)
                    if new_path != cover_path:
                        del contents[cover_path]
                        old, new = cover_href.encode(), new_href.encode()
                        for name in list(contents):
                            if name.endswith((".xhtml", ".html", ".ncx")) and old in contents[name]:
                                contents[name] = contents[name].replace(old, new)
                    contents[opf_name] = _repair_cover_manifest_entry(
                        contents[opf_name], cover_href, new_href, new_mime
                    )
                    contents[new_path] = new_bytes
                    changed = True

    if not changed:
        return {
            "changed": False,
            "issues_before": issues_before,
            "issues_after": issues_before,
            "unfixable": unfixable,
        }

    backup_path = path + ".bak"
    shutil.copy2(path, backup_path)

    final_names = list(contents.keys())
    if "mimetype" in final_names:
        final_names.remove("mimetype")
        final_names.insert(0, "mimetype")

    tmp_path = path + ".tmp"
    with zipfile.ZipFile(tmp_path, "w") as zf:
        for name in final_names:
            # mimetype must be first and stored uncompressed per the EPUB spec.
            compress = zipfile.ZIP_STORED if name == "mimetype" else zipfile.ZIP_DEFLATED
            zf.writestr(name, contents[name], compress_type=compress)
    os.replace(tmp_path, path)

    issues_after = validate_epub_report(path)
    return {
        "changed": True,
        "issues_before": issues_before,
        "issues_after": issues_after,
        "unfixable": unfixable,
        "backup": os.path.basename(backup_path),
    }


def build_epub(
    title: str,
    author: str,
    ocr_text: str,
    output_path: str,
    cover_image_bytes: bytes | None = None,
    cover_image_mime: str = "image/jpeg",
    images: dict[str, bytes] | None = None,
):
    """
    Assemble and write an EPUB file.

    Args:
        title: Book title (derived from filename if not supplied).
        author: Author string.
        ocr_text: Full markdown text from OCR.
        output_path: Where to write the .epub file.
        cover_image_bytes: Raw bytes of the cover image (None = no cover).
        cover_image_mime: MIME type of cover image (unused; the image is
            re-encoded to JPEG regardless of source format — see
            _prepare_cover).
        images: Package-relative path -> bytes for images the OCR markdown
            references (see extract_pdf_images_for_markdown). <img> tags
            pointing anywhere else are stripped rather than shipped broken.
    """
    ocr_text = _fix_text_encoding(ocr_text)
    images = images or {}
    allowed_image_srcs = frozenset(images)

    book = epub.EpubBook()
    book.set_identifier(str(uuid.uuid4()))
    book.set_title(title)
    book.set_language("en")
    book.add_author(author)

    for i, (image_name, image_bytes) in enumerate(sorted(images.items())):
        ext = image_name.rsplit(".", 1)[-1].lower()
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                "gif": "image/gif", "webp": "image/webp"}.get(ext, "image/jpeg")
        book.add_item(epub.EpubItem(
            uid=f"img_{i:04d}",
            file_name=image_name,
            media_type=mime,
            content=image_bytes,
        ))

    # Cover image
    if cover_image_bytes:
        prepared = _prepare_cover(cover_image_bytes)
        if prepared:
            cover_bytes, cover_filename, _ = prepared
            book.set_cover(cover_filename, cover_bytes)
        else:
            logger.warning("Building EPUB without a cover (validation failed)")

    # CSS
    css = epub.EpubItem(
        uid="style",
        file_name="style/main.css",
        media_type="text/css",
        content=b"""
body { font-family: Georgia, serif; line-height: 1.6; margin: 1em 2em; }
h1, h2, h3 { margin-top: 1.5em; }
p { margin: 0.5em 0; text-indent: 1.5em; }
img { max-width: 100%; height: auto; }
pre, code { font-family: monospace; background: #f4f4f4; padding: 0.2em 0.4em; }
pre { white-space: pre-wrap; overflow-wrap: break-word; }
code { overflow-wrap: break-word; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #ccc; padding: 0.4em 0.8em; overflow-wrap: break-word; }
""",
    )
    book.add_item(css)

    # Split into chapters
    chapters = _split_oversized(_split_chapters(ocr_text))
    epub_chapters = []

    for idx, (chapter_title, chapter_md) in enumerate(chapters):
        html_body = markdown.markdown(
            chapter_md,
            extensions=["tables", "fenced_code"],
        )
        html_body = _sanitize_html_fragment(html_body, chapter_title, allowed_image_srcs)
        safe_title = html.escape(chapter_title, quote=False)
        chapter = epub.EpubHtml(
            title=chapter_title,
            file_name=f"chapter_{idx + 1:03d}.xhtml",
            lang="en",
        )
        # NOTE: do NOT prepend an <?xml ...?> prolog here. ebooklib emits its own
        # XML declaration on write, and a leading prolog makes its
        # get_body_content() parser return an empty body, which later crashes the
        # page-break scan with lxml "Document is empty".
        chapter.content = (
            f"<html xmlns='{XHTML_NS}'>"
            f"<head><title>{safe_title}</title>"
            f"<link rel='stylesheet' type='text/css' href='style/main.css'/>"
            f"</head><body><h1>{safe_title}</h1>{html_body}</body></html>"
        )
        chapter.add_item(css)
        book.add_item(chapter)
        epub_chapters.append(chapter)

    # Navigation
    book.toc = epub_chapters
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav"] + epub_chapters

    epub.write_epub(output_path, book)
    _validate_epub(output_path)
    logger.info("EPUB written to %s", output_path)
