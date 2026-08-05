import unittest

from pydantic import ValidationError

from app.main import ProcessRequest


class ProcessRequestTests(unittest.TestCase):
    def test_reader_safe_is_the_api_default(self):
        request = ProcessRequest(job_id="job-1")
        self.assertEqual(request.mode, "reader_safe")

    def test_full_mode_is_explicitly_supported(self):
        request = ProcessRequest(job_id="job-1", mode="full")
        self.assertEqual(request.mode, "full")

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValidationError):
            ProcessRequest(job_id="job-1", mode="fast")


if __name__ == "__main__":
    unittest.main()
