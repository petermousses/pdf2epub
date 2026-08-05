import tempfile
import unittest
from pathlib import Path

import fitz

from app.ocr import pdf_to_markdown


class PdfTextLayerTests(unittest.TestCase):
    def test_text_layer_becomes_markdown_with_toc(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "text.pdf"
            document = fitz.open()
            page = document.new_page()
            page.insert_text((72, 72), "w = r = (n + 1) / 2")
            document.set_toc([[1, "Intro", 1]])
            document.save(path)
            document.close()

            markdown = pdf_to_markdown(str(path))
            self.assertIn("# Intro", markdown)
            self.assertIn("w = r = (n + 1) / 2", markdown)


if __name__ == "__main__":
    unittest.main()
