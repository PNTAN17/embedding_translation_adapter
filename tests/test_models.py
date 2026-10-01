import unittest

import torch

from embedding_transfer.adapter import EmbeddingTransferAdapter
from embedding_transfer.stance import TextImageStance


class ModelShapeTests(unittest.TestCase):
    def test_adapter_shape(self):
        model = EmbeddingTransferAdapter(
            input_embedding_dim=8,
            output_embedding_dim=6,
            hidden_dim=32,
            num_encoder_layers=2,
            num_decoder_layers=1,
            num_heads=4,
            dropout=0.0,
            max_seq_length=7,
        ).eval()
        source = torch.randn(2, 7, 8)
        mask = torch.tensor([[1] * 7, [1, 1, 1, 1, 0, 0, 0]])
        self.assertEqual(tuple(model(source, mask).shape), (2, 7, 6))

    def test_stance_shapes(self):
        model = TextImageStance().eval()
        claim = torch.randn(2, 4, 512)
        claim_mask = torch.ones(2, 4, dtype=torch.long)
        evidence = torch.randn(2, 3, 5, 512)
        evidence_mask = torch.ones(2, 3, 5, dtype=torch.long)
        evidence_count = torch.tensor([3, 2])
        images = torch.randn(2, 2, 50, 768)
        image_count = torch.tensor([2, 1])
        self.assertEqual(
            tuple(
                model(
                    claim,
                    claim_mask,
                    evidence,
                    evidence_mask,
                    evidence_count,
                    images,
                    image_count,
                ).shape
            ),
            (2, 3),
        )


if __name__ == "__main__":
    unittest.main()
