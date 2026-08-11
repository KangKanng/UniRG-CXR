import tempfile
import unittest
from pathlib import Path

from main import load_r2gen_splits, stable_split, target, write_jsonl


class PipelineTest(unittest.TestCase):
    def test_split_is_stable(self):
        self.assertEqual(stable_split("123"), stable_split("123"))

    def test_r2gen_split_counts(self):
        splits = load_r2gen_splits(Path("data/rexrank"))
        self.assertEqual(len(splits), 2955)
        self.assertEqual(sum(x == "train" for x in splits.values()), 2069)
        self.assertEqual(sum(x == "valid" for x in splits.values()), 296)
        self.assertEqual(sum(x == "test" for x in splits.values()), 590)

    def test_jsonl_written(self):
        row = {"id": "1", "split": "train", "image": "/x.png", "prompt": "p", "answer": target("f", "i")}
        with tempfile.TemporaryDirectory() as d:
            write_jsonl([row], Path(d))
            self.assertIn('"messages"', (Path(d) / "iu_train.jsonl").read_text())


if __name__ == "__main__":
    unittest.main()
