"""Zero-mention handling: preprocessing marks them, inference fills them in."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from coref.data.conll_parser import Document, Mention, Sentence, Token
from coref.data.preprocessor import create_frame_examples
from coref.eval.inference import (reconstruct_output, run_inference_with_predictor,
                                  split_masked_input)


def _two_sentence_doc():
    """'He said <zero> hello .' / 'She left .' with He and the zero mention
    in the same cluster."""
    sent1 = Sentence(0, [Token(0, "He", "-"), Token(1, "said", "-"),
                         Token(2, "hello", "-"), Token(3, ".", "-")])
    sent2 = Sentence(1, [Token(0, "She", "-"), Token(1, "left", "-"),
                         Token(2, ".", "-")])
    overt = Mention(0, 0, 0, 0, is_zero=False)
    zero  = Mention(0, 2, 2, 0, is_zero=True)
    return Document("doc1", "en", [sent1, sent2], [overt, zero], {0: [overt, zero]})


class TestZeroMentions(unittest.TestCase):

    def test_preprocessor_marks_zero_mentions_differently(self):
        doc = _two_sentence_doc()
        # Small frame budget so the two sentences land in separate frames.
        examples = create_frame_examples(doc, instruction_id=5, max_tokens_per_frame=10)
        self.assertTrue(examples)

        ex = examples[0]
        # Overt mentions use '#MASK', zero mentions '@MASK'.
        self.assertIn("<m>He</m>#MASK", ex.masked_input)
        self.assertIn("</z>@MASK", ex.masked_input)
        # Both are filled with the same local number in the target output,
        # because both belong to cluster 0.
        self.assertIn("<m>He</m>#0", ex.output)
        self.assertIn("</z>#0", ex.output)

    def test_both_mask_kinds_are_split_points(self):
        masked = "<m>He</m>#MASK said </z>@MASK hello ."
        segments = split_masked_input(masked)
        self.assertEqual(len(segments) - 1, 2)

    def test_reconstruction_uses_hash_for_both_kinds(self):
        masked = "<m>He</m>#MASK said </z>@MASK hello ."
        segments = split_masked_input(masked)
        self.assertEqual(
            reconstruct_output(segments, [0, 0]),
            "<m>He</m>#0 said </z>#0 hello .",
        )

    def test_predictions_attach_to_mentions_in_order(self):
        doc = _two_sentence_doc()
        frames = create_frame_examples(doc, instruction_id=5, max_tokens_per_frame=10)

        results = run_inference_with_predictor(
            frames[:1], lambda instruction, masked, n: [0] * n
        )
        result = results[0]

        mentions = result["before_mentions"] + result["after_mentions"]
        self.assertEqual([m["predicted_local_no"] for m in mentions], [0, 0])
        self.assertIn("</z>#0", result["output_text"])

    def test_missing_predictions_become_fresh_singletons(self):
        """A model that stops early must not have its gaps merged into cluster 0."""
        doc = _two_sentence_doc()
        frames = create_frame_examples(doc, instruction_id=5, max_tokens_per_frame=10)

        results = run_inference_with_predictor(
            frames[:1], lambda instruction, masked, n: [0]      # only one answer
        )
        mentions = results[0]["before_mentions"] + results[0]["after_mentions"]
        predicted = [m["predicted_local_no"] for m in mentions]

        self.assertEqual(predicted[0], 0)
        self.assertNotIn(0, predicted[1:])          # the gap got its own cluster
        self.assertEqual(len(set(predicted)), len(predicted))


if __name__ == "__main__":
    unittest.main()
