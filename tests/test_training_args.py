"""
Tests for training-argument construction (coref/modeling/train.py).

Hyperparameters reach the trainer through a class whose accepted keywords
change between library versions. A keyword the installed class does not accept
is dropped, which means a setting can stop applying without the run failing —
transformers 5 removing `warmup_ratio` is exactly that case. These tests pin the
behaviour that keeps such a change visible.

Skipped when transformers is not installed (the CPU-only validation path).
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import transformers  # noqa: F401
    _HAVE_TRANSFORMERS = True
except ImportError:
    _HAVE_TRANSFORMERS = False


@unittest.skipUnless(_HAVE_TRANSFORMERS, "transformers not installed")
class TestBuildTrainingArgs(unittest.TestCase):

    def _build(self, **overrides):
        from coref.modeling.train import _build_training_args
        kwargs = dict(
            output_dir=self.tmp,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            warmup_ratio=0.03,
            warmup_steps=77,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            # train() disables both on a CPU-only host; mirror that here, or
            # TrainingArguments refuses to construct at all.
            bf16=False,
            fp16=False,
            seed=42,
            report_to="none",
            max_seq_length=4096,
            dataset_text_field="text",
        )
        kwargs.update(overrides)
        return _build_training_args(**kwargs)

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_warmup_survives_whichever_spelling_is_accepted(self):
        """Both spellings are offered; one must land. A run with no warmup at
        lr 2e-4 is not the configured run."""
        args = self._build()
        applied = getattr(args, "warmup_steps", 0) or getattr(args, "warmup_ratio", 0)
        self.assertTrue(applied, "neither warmup_steps nor warmup_ratio applied")

    def test_core_hyperparameters_are_not_dropped(self):
        args = self._build()
        self.assertEqual(args.num_train_epochs, 3)
        self.assertEqual(args.per_device_train_batch_size, 4)
        self.assertEqual(args.gradient_accumulation_steps, 4)
        self.assertAlmostEqual(args.learning_rate, 2e-4)
        self.assertAlmostEqual(args.weight_decay, 0.01)
        self.assertEqual(args.seed, 42)

    def test_sequence_length_reaches_the_config(self):
        """max_seq_length was renamed max_length in trl 0.20; losing it means
        silent truncation at the library default."""
        args = self._build()
        length = getattr(args, "max_length", None) or getattr(args, "max_seq_length", None)
        self.assertEqual(length, 4096)

    def test_unknown_keyword_does_not_raise(self):
        """Forward compatibility: an argument a future version drops must
        degrade, not crash the run at startup."""
        args = self._build(definitely_not_a_real_argument=123)
        self.assertEqual(args.num_train_epochs, 3)


if __name__ == "__main__":
    unittest.main()
