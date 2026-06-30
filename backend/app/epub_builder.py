"""
Build an EPUB from OCR-extracted markdown text and an optional cover image.
"""

import io
import re
import uuid
import logging
from pathlib import Path

import markdown
from ebooklib import epub

logger = logging.getLogger(__name__)


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
        cover_image_mime: MIME type of cover image.
    """
    book = epub.EpubBook()
    book.set_identifier(str(uuid.uuid4()))
    book.set_title(title)
    book.set_language("en")
    book.add_author(author)

    # Cover image
    if cover_image_bytes:
        ext = "jpg" if "jpeg" in cover_image_mime else cover_image_mime.split("/")[-1]
        cover_filename = f"cover.{ext}"
        book.set_cover(cover_filename, cover_image_bytes)

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
        chapter = epub.EpubHtml(
            title=chapter_title,
            file_name=f"chapter_{idx + 1:03d}.xhtml",
            lang="en",
        )
        chapter.content = (
            f"<?xml version='1.0' encoding='UTF-8'?>"
            f"<html xmlns='http://www.w3.org/1999/xhtml'>"
            f"<head><title>{chapter_title}</title>"
            f"<link rel='stylesheet' type='text/css' href='style/main.css'/>"
            f"</head><body><h1>{chapter_title}</h1>{html_body}</body></html>"
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
    logger.info("EPUB written to %s", output_path)
