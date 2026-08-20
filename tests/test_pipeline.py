"""
Regression tests for the parts of the pipeline that need no GPU and no torch:
CoNLL parsing, frame construction, Algorithm 1, and the CoNLL scorer.

Run with:  python -m unittest discover -s tests
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from coref.data.conll_parser import Sentence, Token, parse_conll_file
from coref.eval.evaluate import b3_score, ceafe_score, evaluate_documents, muc_score
from coref.eval.postprocessor import merge_clusters_over_frames


def _write_conll(body: str) -> str:
    fh = tempfile.NamedTemporaryFile("w", suffix=".conll", delete=False,
                                     encoding="utf-8")
    fh.write(body)
    fh.close()
    return fh.name


def _row(doc, part, idx, word, coref):
    return f"{doc}\t{part}\t{idx}\t{word}\tX\t*\t-\t-\t-\t-\t*\t*\t*\t*\t*\t*\t{coref}"


class TestConllParser(unittest.TestCase):

    def test_parts_become_separate_documents(self):
        """Parts restart sentence and cluster numbering, so they must not share
        a doc_id — otherwise their mention keys collide and clusters merge."""
        body = "\n".join([
            "#begin document (d1); part 0",
            _row("d1", 0, 0, "A", "(0)"),
            "",
            "#end document",
            "#begin document (d1); part 1",
            _row("d1", 1, 0, "B", "(0)"),
            "",
            "#end document",
            "",
        ])
        path = _write_conll(body)
        try:
            docs = parse_conll_file(path, language="hi")
        finally:
            os.unlink(path)

        self.assertEqual(len(docs), 2)
        self.assertNotEqual(docs[0].doc_id, docs[1].doc_id)
        self.assertEqual([d.part for d in docs], [0, 1])
        self.assertEqual([d.base_doc_id for d in docs], ["d1", "d1"])

    def test_nested_same_cluster_spans(self):
        """'(3(3 … 3)3)' must yield two mentions, not one."""
        body = "\n".join([
            "#begin document (d2); part 0",
            _row("d2", 0, 0, "the", "(3"),
            _row("d2", 0, 1, "man", "(3"),
            _row("d2", 0, 2, "s", "3)"),
            _row("d2", 0, 3, "dog", "3)"),
            "",
            "#end document",
            "",
        ])
        path = _write_conll(body)
        try:
            doc = parse_conll_file(path)[0]
        finally:
            os.unlink(path)

        spans = sorted((m.start_tok, m.end_tok) for m in doc.mentions)
        self.assertEqual(spans, [(0, 3), (1, 2)])

    def test_unclosed_span_does_not_leak_into_next_sentence(self):
        """An unclosed '(' must be closed at the end of its own sentence."""
        body = "\n".join([
            "#begin document (d3); part 0",
            _row("d3", 0, 0, "A", "(7"),
            _row("d3", 0, 1, "B", "-"),
            "",
            _row("d3", 0, 0, "C", "-"),
            _row("d3", 0, 1, "D", "-"),
            "",
            "#end document",
            "",
        ])
        path = _write_conll(body)
        try:
            doc = parse_conll_file(path)[0]
        finally:
            os.unlink(path)

        self.assertEqual(len(doc.mentions), 1)
        m = doc.mentions[0]
        self.assertEqual((m.sent_idx, m.start_tok, m.end_tok), (0, 0, 1))


class TestAlgorithm1(unittest.TestCase):

    @staticmethod
    def _mention(sent, tok, local):
        return {"sent_idx": sent, "start_tok": tok, "end_tok": tok,
                "predicted_local_no": local}

    def test_cluster_is_carried_across_the_frame_overlap(self):
        """Frame 1 is the 'after' of pair 0 and the 'before' of pair 1; a cluster
        shared through it must keep one global id."""
        m = self._mention
        results = [
            {"before_mentions": [m(0, 0, 0)], "after_mentions": [m(1, 0, 0)]},
            # Same mention (1,0,0) reappears as 'before', now numbered 2 locally.
            {"before_mentions": [m(1, 0, 2)], "after_mentions": [m(2, 0, 2)]},
        ]
        glob, clusters = merge_clusters_over_frames(results)

        self.assertEqual(glob[(0, 0, 0)], glob[(1, 0, 0)])
        self.assertEqual(glob[(1, 0, 0)], glob[(2, 0, 0)])
        self.assertEqual(len(clusters), 1)

    def test_unseen_local_number_gets_a_fresh_global_id(self):
        m = self._mention
        results = [
            {"before_mentions": [m(0, 0, 0)], "after_mentions": [m(1, 0, 1)]},
        ]
        glob, clusters = merge_clusters_over_frames(results)

        self.assertNotEqual(glob[(0, 0, 0)], glob[(1, 0, 0)])
        self.assertEqual(len(clusters), 2)


class TestScorer(unittest.TestCase):

    def test_perfect_prediction_scores_100(self):
        gold = {0: {(0, 0, 0), (0, 1, 1)}, 1: {(1, 0, 0), (1, 1, 1)}}
        scores = evaluate_documents([gold], [dict(gold)])
        for metric in ("muc", "b3", "ceafe", "conll"):
            self.assertAlmostEqual(scores[metric]["f"], 100.0, places=2)

    def test_muc_precision_and_recall_differ_when_over_merged(self):
        """Merging two gold clusters gives perfect recall but not precision.
        With a shared numerator (the old bug) both came out equal."""
        gold = {0: {(0, 0, 0), (0, 1, 1)}, 1: {(0, 2, 2), (0, 3, 3)}}
        pred = {0: {(0, 0, 0), (0, 1, 1), (0, 2, 2), (0, 3, 3)}}

        p, r, f = muc_score(gold, pred)
        self.assertAlmostEqual(r, 1.0, places=6)      # every gold link recovered
        self.assertAlmostEqual(p, 2 / 3, places=6)    # 2 of 3 predicted links right
        self.assertLess(p, r)

    def test_all_singletons_has_zero_muc_recall(self):
        gold = {0: {(0, 0, 0), (0, 1, 1)}}
        pred = {0: {(0, 0, 0)}, 1: {(0, 1, 1)}}

        p, r, f = muc_score(gold, pred)
        self.assertEqual(r, 0.0)
        self.assertEqual(f, 0.0)

    def test_b3_and_ceafe_penalise_over_merging(self):
        gold = {0: {(0, 0, 0), (0, 1, 1)}, 1: {(0, 2, 2), (0, 3, 3)}}
        pred = {0: {(0, 0, 0), (0, 1, 1), (0, 2, 2), (0, 3, 3)}}

        b3_p, b3_r, _ = b3_score(gold, pred)
        self.assertAlmostEqual(b3_r, 1.0, places=6)
        self.assertAlmostEqual(b3_p, 0.5, places=6)

        ce_p, ce_r, _ = ceafe_score(gold, pred)
        self.assertLess(ce_r, 1.0)   # one pred entity cannot match two gold ones

    def test_micro_average_matches_single_doc_score(self):
        gold = {0: {(0, 0, 0), (0, 1, 1)}, 1: {(0, 2, 2), (0, 3, 3)}}
        pred = {0: {(0, 0, 0), (0, 1, 1), (0, 2, 2)}, 1: {(0, 3, 3)}}

        single = evaluate_documents([gold], [pred])
        p, r, f = muc_score(gold, pred)
        self.assertAlmostEqual(single["muc"]["f"], round(f * 100, 2), places=2)


if __name__ == "__main__":
    unittest.main()


class TestDuplicateSpanAnnotations(unittest.TestCase):
    """The corpus annotates some spans for two clusters at once ('(3|(4' …
    '3)|4)'). A span carries one position key and the model emits one number
    per mask, so the surplus annotation has to be resolved at parse time — and
    resolved the same way every time, or gold and predicted clusterings
    disagree about a span that was linked correctly."""

    def _doc_with_double_annotation(self):
        body = "\n".join([
            "#begin document (d1); part 0",
            _row("d1", 0, 0, "both", "(3|(4"),
            _row("d1", 0, 1, "countries", "3)|4)"),
            _row("d1", 0, 2, "they", "(3)"),
            "",
            "#end document",
        ]) + "\n"
        path = _write_conll(body)
        try:
            return parse_conll_file(path)[0]
        finally:
            os.unlink(path)

    def test_span_keeps_exactly_one_annotation(self):
        doc = self._doc_with_double_annotation()
        keys = [m.position_key for m in doc.mentions]
        self.assertEqual(len(keys), len(set(keys)),
                         "a span must not appear twice in doc.mentions")

    def test_lowest_cluster_id_wins_deterministically(self):
        doc = self._doc_with_double_annotation()
        span = [m for m in doc.mentions if m.position_key == (0, 0, 1)]
        self.assertEqual(len(span), 1)
        self.assertEqual(span[0].cluster_id, 3)

    def test_other_mentions_are_untouched(self):
        doc = self._doc_with_double_annotation()
        self.assertIn((0, 2, 2), [m.position_key for m in doc.mentions])

    def test_clusters_index_agrees_with_mention_list(self):
        """doc.clusters is built from the deduplicated list, so the surviving
        span must appear in cluster 3 and nowhere else."""
        doc = self._doc_with_double_annotation()
        in_4 = [m.position_key for m in doc.clusters.get(4, [])]
        self.assertNotIn((0, 0, 1), in_4)
        self.assertIn((0, 0, 1), [m.position_key for m in doc.clusters[3]])
