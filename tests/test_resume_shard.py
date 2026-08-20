"""
Tests for inference resume (scripts/run_inference.py `_load_shard`).

A full test-split inference run is hours long and will be interrupted. What
must hold is that an interrupted run loses at most the document it was working
on — including when the process died mid-write and left a half-written line.

Pure Python — no torch, no model.
"""

import importlib.util
import json
import os
import sys
import tempfile
import unittest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)


def _load_run_inference():
    """Import the script by path — scripts/ is not a package."""
    path = os.path.join(_ROOT, "scripts", "run_inference.py")
    spec = importlib.util.spec_from_file_location("run_inference_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run_inference = _load_run_inference()


def _record(doc_id, lang="hi"):
    return {"doc_id": doc_id, "language": lang,
            "gold": {"0": [[0, 0, 1]]}, "pred": {"0": [[0, 0, 1]]}}


class TestLoadShard(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "predictions_test.jsonl")

    def _write(self, text):
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write(text)

    def test_missing_file_is_an_empty_start(self):
        self.assertEqual(run_inference._load_shard(self.path), {})

    def test_completed_documents_are_returned_by_id(self):
        self._write("\n".join(json.dumps(_record(d)) for d in ("a", "b")) + "\n")
        done = run_inference._load_shard(self.path)
        self.assertEqual(sorted(done), ["a", "b"])
        self.assertEqual(done["a"]["language"], "hi")

    def test_truncated_final_line_keeps_everything_before_it(self):
        """The process was killed mid-write. The 2 complete documents must
        survive; only the partial one is re-run."""
        good = "\n".join(json.dumps(_record(d)) for d in ("a", "b"))
        self._write(good + "\n" + '{"doc_id": "c", "lang')
        done = run_inference._load_shard(self.path)
        self.assertEqual(sorted(done), ["a", "b"])

    def test_blank_lines_are_skipped(self):
        self._write(json.dumps(_record("a")) + "\n\n" + json.dumps(_record("b")) + "\n")
        self.assertEqual(sorted(run_inference._load_shard(self.path)), ["a", "b"])

    def test_rerun_of_a_document_takes_the_later_record(self):
        """Appending is how the shard grows, so a doc re-run after --no-resume
        was forgotten appears twice; the newest write wins."""
        first = _record("a"); first["language"] = "hi"
        second = _record("a"); second["language"] = "ta"
        self._write(json.dumps(first) + "\n" + json.dumps(second) + "\n")
        done = run_inference._load_shard(self.path)
        self.assertEqual(done["a"]["language"], "ta")


class TestSafeName(unittest.TestCase):

    def test_path_separators_and_hashes_are_stripped(self):
        """doc_ids come from '#begin document' lines and contain / and #;
        unescaped they would write outside the output directory."""
        out = run_inference._safe_name("wb/a2e/00/a2e_0000#1")
        self.assertNotIn("/", out)
        self.assertNotIn("#", out)


if __name__ == "__main__":
    unittest.main()
