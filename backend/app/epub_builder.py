"""
Build an EPUB from OCR-extracted markdown text and an optional cover image.
"""

import html
import io
import os
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


def _split_chapters(text: str) -> list[tuple[str, str]]:
    """
    Split markdown text into (title, content) chapters at H1/H2 headings.
    Falls back to a single chapter if no headings found.
    """
    # Split on lines that start with # or ##
    chapter_re = re.compile(r"^(#{1,2})\s+(.+)$", re.MULTILINE)
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


# A tag whose name carries a namespace prefix, e.g. <xsl:template match="…">
# or </fo:block>. python-markdown never emits prefixed elements, so one in
# its output is always OCR'd literal text (an XML/XSLT code sample in the
# source book) passed through as raw HTML.
_PREFIXED_TAG_RE = re.compile(r"</?[A-Za-z][\w.-]*:[^<>]*>")


def _sanitize_html_fragment(body: str, chapter_title: str) -> str:
    """
    Guarantee an HTML fragment produced by python-markdown is well-formed XML.

    OCR'd source text sometimes contains stray '<'/'>' sequences (generics
    like "List<Item>", comparisons like "a<b", math/footnote notation) that
    python-markdown treats as raw HTML passthrough and leaves unescaped and
    unclosed. The resulting XHTML file is not well-formed XML. EPUB content
    documents are XML, and strict e-reader firmware parses them as such —
    it will fail (or hang) loading a chapter that isn't well-formed, even
    though lenient tools like browsers or Calibre tolerate it. Detect that
    case and repair it with a lenient HTML parser so the shipped file is
    always valid XML.
    """
    probe_template = f"<div xmlns='{XHTML_NS}'>{{}}</div>"
    try:
        etree.fromstring(probe_template.format(body).encode("utf-8"))
        return body
    except etree.XMLSyntaxError as e:
        logger.warning(
            "Chapter %r: markdown output was not well-formed XML (%s); repairing",
            chapter_title,
            e,
        )

    # Namespace-prefixed tags survive the lenient HTML reparse below as
    # elements, but XML requires their prefix to be declared, so the
    # "repaired" fragment would still be rejected by every XML parser.
    # Since they can only be code-sample text, escape them so they render
    # as the visible text the book intended.
    escaped = _PREFIXED_TAG_RE.sub(
        lambda m: html.escape(m.group(0), quote=False), body
    )

    fragment = lhtml.fragment_fromstring(escaped, create_parent="div")
    repaired = etree.tostring(fragment, encoding="unicode", method="xml")
    repaired = repaired[len("<div>") : -len("</div>")]
    try:
        etree.fromstring(probe_template.format(repaired).encode("utf-8"))
        return repaired
    except etree.XMLSyntaxError as e:
        # The lenient parse can still let non-XML constructs through. Ship
        # the chapter as escaped preformatted text rather than an EPUB that
        # fails validation.
        logger.warning(
            "Chapter %r: repair still not well-formed XML (%s); "
            "falling back to escaped text",
            chapter_title,
            e,
        )
        return "<pre>{}</pre>".format(html.escape(body, quote=False))


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
    re-serialize any content document that isn't well-formed XML, and
    re-encode the cover image as a flattened baseline JPEG if it isn't one
    already (see validate_epub_report). Assumes this app's own package
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

    # 1. Repair malformed XHTML/HTML content documents.
    for name in list(contents):
        if not name.endswith((".xhtml", ".html")):
            continue
        data = contents[name]
        try:
            etree.fromstring(data)
            continue
        except etree.XMLSyntaxError:
            pass
        try:
            repaired = _reparse_as_xhtml(data)
            etree.fromstring(repaired)  # verify the repair actually took
        except Exception as e:
            unfixable.append(f"{name}: could not be repaired ({e})")
            continue
        contents[name] = repaired
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
    """
    book = epub.EpubBook()
    book.set_identifier(str(uuid.uuid4()))
    book.set_title(title)
    book.set_language("en")
    book.add_author(author)

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
pre, code { font-family: monospace; background: #f4f4f4; padding: 0.2em 0.4em; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #ccc; padding: 0.4em 0.8em; }
""",
    )
    book.add_item(css)

    # Split into chapters
    chapters = _split_chapters(ocr_text)
    epub_chapters = []

    for idx, (chapter_title, chapter_md) in enumerate(chapters):
        html_body = markdown.markdown(
            chapter_md,
            extensions=["tables", "fenced_code"],
        )
        html_body = _sanitize_html_fragment(html_body, chapter_title)
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
    try:
        _validate_epub(output_path)
    except Exception:
        # Don't leave a broken EPUB in the output directory — it would show
        # up in the library looking like a finished book.
        try:
            os.remove(output_path)
        except OSError:
            pass
        raise
    logger.info("EPUB written to %s", output_path)
