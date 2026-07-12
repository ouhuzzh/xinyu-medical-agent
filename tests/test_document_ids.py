import sys
import unittest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "project"))

from db.document_ids import build_document_no  # noqa: E402


class DocumentIdTests(unittest.TestCase):
    def test_build_document_no_keeps_short_stem(self):
        self.assertEqual(build_document_no("short-name.pdf"), "short-name")

    def test_build_document_no_truncates_long_stem_stably(self):
        source_name = "medlineplus-afecciones-posteriores-al-covid-19-covid-19-persistente-muy-largo很长名字.pdf"
        first = build_document_no(source_name)
        second = build_document_no(source_name)

        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 64)
        self.assertIn("-", first)


if __name__ == "__main__":
    unittest.main()
