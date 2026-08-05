import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from app.epub_builder import (
    _render_math_in_markdown,
    _sanitize_html_fragment,
    _split_chapters,
    _split_oversized,
    build_epub,
    validate_epub_report,
)


def _jpeg_bytes() -> bytes:
    image = Image.new("RGB", (12, 12), (40, 80, 120))
    output = io.BytesIO()
    image.save(output, format="JPEG")
    return output.getvalue()


class EpubBuilderTests(unittest.TestCase):
    def test_math_markers_render_as_xhtml_subscripts_and_superscripts(self):
        rendered = _render_math_in_markdown(
            r"The relation is σ_{family} and the value is x^{2}."
        )
        self.assertIn('<span class="math" role="math">σ<sub>family</sub></span>', rendered)
        self.assertIn('<span class="math" role="math">x<sup>2</sup></span>', rendered)

    def test_compact_pdf_text_layer_subscript_is_recovered(self):
        rendered = _render_math_in_markdown("sharks = σfamily = Sharks")
        self.assertIn('<span class="math" role="math">σ<sub>family</sub></span>', rendered)

    def test_fenced_code_is_not_treated_as_math(self):
        source = "```text\nrecord_{id}\n```"
        self.assertEqual(_render_math_in_markdown(source), source)

    def test_unknown_ocr_tags_are_unwrapped_and_image_refs_are_checked(self):
        body = '<p>before</p><page>garbage</page><img src="images/missing.jpg">'
        sanitized = _sanitize_html_fragment(body, "Test", frozenset())
        self.assertNotIn("<page", sanitized)
        self.assertNotIn("missing.jpg", sanitized)
        self.assertIn("garbage", sanitized)

    def test_chapters_tolerate_indentation_and_split_large_content(self):
        chapters = _split_chapters("  # First\n\nA\n\n# Second\n\nB")
        self.assertEqual([title for title, _ in chapters], ["First", "Second"])
        parts = _split_oversized([("Book", "one\n\ntwo\n\nthree")], max_chars=5)
        self.assertEqual(len(parts), 3)

    def test_build_packages_figures_and_math_and_validates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "book.epub"
            build_epub(
                title="Test",
                author="Tester",
                ocr_text="# Intro\n\nσ_{family} = Sharks\n\n![Figure](images/page_0_0.jpg)",
                output_path=str(output),
                images={"images/page_0_0.jpg": _jpeg_bytes()},
            )

            self.assertEqual(validate_epub_report(str(output)), [])
            with zipfile.ZipFile(output) as archive:
                names = set(archive.namelist())
                self.assertTrue(any(name.endswith("page_0_0.jpg") for name in names))
                chapter = next(
                    archive.read(name).decode("utf-8")
                    for name in names
                    if name.endswith("chapter_001.xhtml")
                )
                self.assertIn("<sub>family</sub>", chapter)
                self.assertIn("<img", chapter)


if __name__ == "__main__":
    unittest.main()
