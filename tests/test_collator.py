"""
Tests for completion-only loss masking (coref/modeling/collator.py).

These cover the property the training run depends on and cannot easily observe:
that loss is computed on the assistant's cluster numbers and on nothing else.
If prompt tokens leak into the loss the model is partly trained to regenerate
the instruction, and the run looks healthy the whole way through.

Pure Python — no torch, no tokenizer, no GPU.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from coref.modeling.collator import (IGNORE_INDEX, count_supervised_tokens,
                                     find_subsequence, mask_prompt_labels)

# Stand-ins for token ids: 9, 8 is the "response template", everything else is text.
TEMPLATE = [9, 8]


class TestFindSubsequence(unittest.TestCase):

    def test_returns_index_after_the_match(self):
        self.assertEqual(find_subsequence([1, 2, 9, 8, 5], TEMPLATE), 4)

    def test_absent_needle_returns_none(self):
        self.assertIsNone(find_subsequence([1, 2, 3], TEMPLATE))

    def test_last_occurrence_wins(self):
        """A copy of the template inside the user text must not end the mask
        early — only the final one starts the assistant turn."""
        self.assertEqual(find_subsequence([9, 8, 1, 9, 8, 7], TEMPLATE), 5)

    def test_needle_longer_than_haystack(self):
        self.assertIsNone(find_subsequence([9], TEMPLATE))


class TestMaskPromptLabels(unittest.TestCase):

    def test_prompt_is_masked_and_answer_is_kept(self):
        ids = [1, 2, 3, 9, 8, 4, 5]
        labels = mask_prompt_labels(ids, TEMPLATE)
        self.assertEqual(labels, [IGNORE_INDEX] * 5 + [4, 5])

    def test_template_itself_is_not_supervised(self):
        """The header tokens are part of the prompt; predicting them teaches
        nothing about coreference."""
        ids = [1, 9, 8, 4]
        labels = mask_prompt_labels(ids, TEMPLATE)
        self.assertEqual(labels[:3], [IGNORE_INDEX] * 3)
        self.assertEqual(labels[3], 4)

    def test_padding_after_the_answer_is_masked(self):
        ids = [1, 9, 8, 4, 5, 0, 0]
        labels = mask_prompt_labels(ids, TEMPLATE, pad_token_id=0)
        self.assertEqual(labels, [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX,
                                  4, 5, IGNORE_INDEX, IGNORE_INDEX])

    def test_truncated_example_contributes_nothing(self):
        """An example whose assistant turn was cut off by max_seq_length has no
        target. Masking it entirely is right; training on its prompt is not."""
        ids = [1, 2, 3, 4]
        labels = mask_prompt_labels(ids, TEMPLATE)
        self.assertEqual(labels, [IGNORE_INDEX] * 4)
        self.assertEqual(count_supervised_tokens(labels), 0)

    def test_does_not_mutate_its_input(self):
        ids = [1, 9, 8, 4]
        mask_prompt_labels(ids, TEMPLATE)
        self.assertEqual(ids, [1, 9, 8, 4])

    def test_pad_id_colliding_with_a_real_answer_token(self):
        """pad_token_id is often eos_token_id. Tokens before the answer are
        masked anyway, so only trailing padding should be affected."""
        ids = [1, 9, 8, 0, 5]
        labels = mask_prompt_labels(ids, TEMPLATE, pad_token_id=0)
        # The 0 sits inside the answer span and is masked as padding — accepted
        # cost of pad==eos, and it is why the answer's own eos is not supervised.
        self.assertEqual(labels, [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX,
                                  IGNORE_INDEX, 5])


class TestSupervisedTokenCount(unittest.TestCase):

    def test_counts_only_unmasked(self):
        self.assertEqual(
            count_supervised_tokens([IGNORE_INDEX, 3, 4, IGNORE_INDEX]), 2
        )


if __name__ == "__main__":
    unittest.main()
