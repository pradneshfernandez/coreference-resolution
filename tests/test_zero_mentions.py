
import sys
import os
import unittest

from unittest.mock import MagicMock
import sys

# Mock torch and transformers before they are imported by src.inference
mock_torch = MagicMock()
mock_torch.device = MagicMock()
sys.modules["torch"] = mock_torch

mock_transformers = MagicMock()
sys.modules["transformers"] = mock_transformers
sys.modules["peft"] = MagicMock()

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.conll_parser import Document, Sentence, Token, Mention
from src.preprocessor import create_frame_examples, get_instruction
from src.inference import controlled_inference


class TestZeroMentions(unittest.TestCase):
    def test_preprocessor_zero_mentions(self):
        # Create a dummy document with one overt and one zero mention
        tokens = [
            Token(0, "He", "-"),
            Token(1, "said", "-"),
            Token(2, "hello", "-"),
            Token(3, ".", "-"),
        ]
        sent = Sentence(0, tokens)
        
        # Overt mention: "He" (cluster 0)
        m1 = Mention(0, 0, 0, 0, is_zero=False)
        # Zero mention: after "said" (cluster 0)
        m2 = Mention(0, 2, 2, 0, is_zero=True)
        
        doc = Document("doc1", "en", [sent], [m1, m2], {0: [m1, m2]})
        
        # Create examples
        examples = create_frame_examples(doc, instruction_id=5, max_tokens_per_frame=256)
        
        # We need at least 2 frames for a pair. Since we only have 1 sentence, 
        # create_frame_examples might return empty if it can't split.
        # Let's add a second sentence.
        tokens2 = [Token(0, "She", "-"), Token(1, "left", "-"), Token(2, ".", "-")]
        sent2 = Sentence(1, tokens2)
        doc.sentences.append(sent2)
        
        examples = create_frame_examples(doc, instruction_id=5, max_tokens_per_frame=10)
        self.assertTrue(len(examples) > 0)
        
        ex = examples[0]
        # Check masked input
        # Expected: "<m>He</m>#MASK said </z>@MASK hello . [MID] She left ."
        # Wait, relative indexing: m2 is at token 2 (hello).
        # In _build_masked_sentence: 
        # pos=0: m1 starts here. outer=m1. Append <m>He</m>#MASK. pos=1.
        # pos=1: said.
        # pos=2: m2 starts here. outer=m2 (zero). Append </z>@MASK. pos=2 (still).
        # pos=2: hello.
        self.assertIn("</z>@MASK", ex.masked_input)
        self.assertIn("<m>He</m>#MASK", ex.masked_input)
        
        # Check output
        self.assertIn("</z>#0", ex.output)
        self.assertIn("<m>He</m>#0", ex.output)

    def test_inference_mask_splitting(self):
        # Mocking regex splitting in controlled_inference
        masked_input = "<m>He</m>#MASK said </z>@MASK hello ."
        instruction = "solve"
        
        # Mock model and tokenizer
        class MockTokenizer:
            model_max_length = 512
            eos_token_id = 2
            def __call__(self, text, **kwargs):
                res = MagicMock()
                res.__getitem__.side_effect = lambda k: torch.zeros((1, 10), dtype=torch.long)
                res.to.return_value = res
                # Simulate the dict-like behavior for "input_ids"
                inner_dict = {"input_ids": torch.zeros((1, 10), dtype=torch.long)}
                res.__getitem__.side_effect = inner_dict.__getitem__
                return res
            def decode(self, ids, **kwargs):
                return "0" # Predict cluster 0
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
                return "prompt"


        class MockModel:
            def parameters(self):
                return [torch.nn.Parameter(torch.zeros(1))]
            def eval(self): pass
            def generate(self, **kwargs):
                return torch.zeros((1, 15), dtype=torch.long)

        import torch
        model = MockModel()
        tokenizer = MockTokenizer()
        device = torch.device("cpu")
        
        output, predicted = controlled_inference(model, tokenizer, instruction, masked_input, device)
        
        # Should have found 2 masks
        self.assertEqual(len(predicted), 2)
        self.assertEqual(predicted, [0, 0])
        self.assertEqual(output, "<m>He</m>#0 said </z>#0 hello .")

if __name__ == "__main__":
    unittest.main()
