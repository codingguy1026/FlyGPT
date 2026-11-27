"""Offline smoke tests. No network or actual corpus download is required."""
import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from papiti_lm.corpus import DigitSubwordTokenizer, Example
from scripts.train_papiti_foundation import (
    acceptable, key, read_cc0, train_tokenizer,
)


class FoundationTests(unittest.TestCase):
    def test_cc0_import_and_deduplication(self):
        rows = (
            "1\teng\tThe little robot opened its notebook.\n"
            "2\teng\tTHE LITTLE ROBOT OPENED ITS NOTEBOOK.\n"
            "3\tkor\t작은 로봇이 공책을 펼치고 공부를 시작했어요.\n"
            "4\teng\tThe robot placed a pencil on the desk.\n"
            "5\teng\tPlease visit https://example.com for details.\n"
            "6\tfra\tUn petit robot ouvre son cahier.\n"
        ).encode("utf-8")
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "test.tar.bz2"
            with tarfile.open(archive, "w:bz2") as out:
                member = tarfile.TarInfo("sentences_CC0.csv")
                member.size = len(rows)
                out.addfile(member, io.BytesIO(rows))
            pool = read_cc0(archive, {key("The robot placed a pencil on the desk.")})
        self.assertEqual([r["id"] for r in pool["en"]], ["1"])
        self.assertEqual([r["id"] for r in pool["ko"]], ["3"])
        self.assertEqual(pool["ko"][0]["license"], "CC0-1.0")

    def test_language_filter(self):
        self.assertTrue(acceptable("en", "The little robot opened its notebook."))
        self.assertTrue(acceptable("ko", "작은 로봇이 공책을 펼치고 공부를 시작했어요."))
        self.assertFalse(acceptable("en", "작은 로봇이 공책을 펼치고 공부를 시작했어요."))
        self.assertFalse(acceptable("ko", "The little robot opened its notebook."))

    def test_counted_bpe_matches_existing_algorithm(self):
        rows = [
            Example("test", "Language: en", "The little robot opened its notebook."),
            Example("test", "Language: en", "The little robot opened its notebook again."),
            Example("test", "Language: ko", "작은 로봇이 공책을 펼쳤어요."),
        ] * 3
        expected = DigitSubwordTokenizer.from_examples(rows, max_merges=24)
        got = train_tokenizer(rows, 24)
        self.assertEqual(got.characters, expected.characters)
        self.assertEqual(got.merges, expected.merges)
        for row in rows:
            self.assertEqual(got.sequence(row, 256), expected.sequence(row, 256))


if __name__ == "__main__":
    unittest.main()
