from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from PIL import Image
import torch
from torch import nn
from transformers import BatchEncoding, BlipTextConfig
from transformers.models.blip.modeling_blip_text import BlipTextModel

from embedding_transfer.vivqa import ViVQAPredictor


class ViVQAInterfaceTests(unittest.TestCase):
    def test_blip_embedding_module_is_bypassed_and_padding_stays_masked(self):
        encoder = BlipTextModel(
            BlipTextConfig(
                hidden_size=8,
                encoder_hidden_size=8,
                intermediate_size=16,
                num_attention_heads=2,
                num_hidden_layers=1,
                vocab_size=8,
            )
        ).eval()
        encoder.embeddings.forward = Mock(
            side_effect=AssertionError("Embedding module must be bypassed")
        )
        generate = Mock(return_value=torch.tensor([[1, 2]]))
        model = SimpleNamespace(
            text_encoder=encoder,
            vision_model=lambda **kwargs: SimpleNamespace(
                last_hidden_state=torch.ones(1, 2, 8)
            ),
            text_decoder=SimpleNamespace(
                config=SimpleNamespace(bos_token_id=1), generate=generate
            ),
        )
        processor = SimpleNamespace(
            image_processor=lambda **kwargs: {"pixel_values": torch.zeros(1, 3, 2, 2)},
            tokenizer=SimpleNamespace(sep_token_id=2, pad_token_id=0),
            decode=lambda *args, **kwargs: "two",
        )
        mask = torch.tensor([[1, 1, 0, 0]])
        tokenizer = lambda *args, **kwargs: BatchEncoding(
            {"input_ids": torch.tensor([[1, 2, 0, 0]]), "attention_mask": mask}
        )

        class Adapter(nn.Module):
            max_seq_length = 4

            def forward(self, source, attention_mask):
                return torch.ones_like(source)

        predictor = ViVQAPredictor.from_modules(
            Adapter(),
            model,
            nn.Embedding(8, 8),
            tokenizer,
            processor,
            torch.device("cpu"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            Image.new("RGB", (2, 2)).save(path)
            self.assertEqual(predictor.predict(path, "question"), "two")
        encoder.embeddings.forward.assert_not_called()
        self.assertTrue(
            torch.equal(generate.call_args.kwargs["encoder_attention_mask"], mask)
        )
        self.assertFalse(generate.call_args.kwargs["do_sample"])


if __name__ == "__main__":
    unittest.main()
