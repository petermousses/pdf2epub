"""
Build an EPUB from OCR-extracted markdown text and an optional cover image.
"""

import html
import io
import re
import uuid
import logging
import zipfile
from pathlib import Path

import markdown
import lxml.html as lhtml
from lxml import etree
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
    probe = f"<div xmlns='{XHTML_NS}'>{body}</div>"
    try:
        etree.fromstring(probe.encode("utf-8"))
        return body
    except etree.XMLSyntaxError as e:
        logger.warning(
            "Chapter %r: markdown output was not well-formed XML (%s); repairing",
            chapter_title,
            e,
        )
        fragment = lhtml.fragment_fromstring(body, create_parent="div")
        repaired = etree.tostring(fragment, encoding="unicode", method="xml")
        return repaired[len("<div>") : -len("</div>")]


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


def _validate_epub(output_path: str) -> None:
    """
    Sanity-check the EPUB actually written to disk before declaring success,
    rather than letting a corrupt file reach the user's reader silently.
    """
    with zipfile.ZipFile(output_path) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"EPUB archive is corrupt at member: {bad}")

        names = zf.namelist()
        if "mimetype" not in names:
            raise RuntimeError("EPUB is missing the required 'mimetype' entry")

        for name in names:
            if name.endswith((".xhtml", ".html", ".ncx", ".opf")) or name.endswith(
                "nav.xhtml"
            ):
                try:
                    etree.fromstring(zf.read(name))
                except etree.XMLSyntaxError as e:
                    raise RuntimeError(
                        f"EPUB validation failed: {name} is not well-formed XML ({e})"
                    ) from e
            elif name.endswith((".jpg", ".jpeg", ".png")):
                try:
                    img = Image.open(io.BytesIO(zf.read(name)))
                    img.load()
                except (UnidentifiedImageError, OSError) as e:
                    raise RuntimeError(
                        f"EPUB validation failed: image {name} is not decodable ({e})"
                    ) from e


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
    _validate_epub(output_path)
    logger.info("EPUB written to %s", output_path)
