# Vietnamese Embedding-Space Transfer

Training and inference code with released checkpoints for **Teaching Frozen English
Vision-Language Models to Read Vietnamese via Embedding-Space Translation
Distillation**.

This repository contains compact pipelines for the two tasks used in the paper.

- ViVQA answers a Vietnamese question about one image with a frozen BLIP VQA
  backbone.
- MOCHEG predicts `refuted`, `supported`, or `NEI` from a Vietnamese claim,
  Vietnamese textual evidence, and optional image evidence.

The training entry points learn the adapter from prepared bilingual JSONL
files. Dataset acquisition, translation, and split construction remain outside
this repository.

## Installation

Python 3.10 or newer is recommended. A CUDA GPU is strongly recommended because
the adapter has about 123 million parameters.

```bash
git clone https://github.com/PNTAN17/embedding_translation_adapter.git
cd embedding_translation_adapter
git lfs install
git lfs pull
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On first use, Transformers downloads the frozen upstream weights for mBERT and
the task backbone. Pass `--local-files-only` after those files are cached if the
machine must run offline.

## ViVQA

```bash
python infer_vivqa.py \
  --image /path/to/vivqa.jpg \
  --question "Có bao nhiêu người trong ảnh?"
```

The command writes one JSON object.

```json
{"answer": "<generated answer>"}
```

## MOCHEG

Repeat `--evidence` and `--image` to provide several evidence items. The released
model uses at most the first five items of each type, matching its training
configuration.

```bash
python infer_mocheg.py \
  --claim "Nội dung của tuyên bố cần kiểm chứng" \
  --evidence "Bằng chứng văn bản thứ nhất" \
  --evidence "Bằng chứng văn bản thứ hai" \
  --image /path/to/evidence.jpg
```

Image evidence is optional. Without `--image`, the frozen text-only classifier
head is used.

```json
{
  "label": "supported",
  "probabilities": {
    "refuted": 0.12,
    "supported": 0.76,
    "NEI": 0.12
  },
  "text_evidence_used": 2,
  "images_used": 1
}
```

The probability values above only illustrate the output schema.

## Training

The repository includes clean adapter trainers for ViVQA and MOCHEG. They
accept separate training and held-out validation files and select checkpoints
only from validation alignment loss.

```bash
python train_vivqa.py --help
python train_mocheg.py --help
```

Input schemas, commands, and model-selection behavior are documented in
[`TRAINING.md`](TRAINING.md).

## Checkpoints

| File | Contents | Released run |
|---|---|---|
| `checkpoints/vivqa_adapter.pt` | ViVQA adapter | seed 42, test exact match 0.4585 |
| `checkpoints/mocheg_adapter.pt` | MOCHEG adapter and frozen stance head | seed 42, test micro F1 0.5025 |

These are the retained checkpoints with the strongest validation result among
the three final seeds. The paper reports aggregate results across seeds where
applicable. The values in this table describe the individual released
checkpoints and should not be substituted for those aggregate values.

The checkpoint files are tracked with Git LFS because each is about 0.5 GB.
After downloading them, verify their hashes with:

```bash
cd checkpoints
sha256sum -c SHA256SUMS
```

Checkpoint loading uses PyTorch's restricted `weights_only=True` mode. You
should still download model artifacts only from a source you trust.

## Repository layout

```text
.
├── checkpoints/                 released weights and SHA-256 hashes
├── embedding_transfer/          shared model, loss, data, and task modules
├── tests/                       lightweight architecture tests
├── infer_vivqa.py               ViVQA command line entry point
├── infer_mocheg.py              MOCHEG command line entry point
├── train_vivqa.py               ViVQA adapter trainer
├── train_mocheg.py              MOCHEG adapter trainer
├── TRAINING.md                  input schemas and training instructions
├── MODEL_CARD.md                scope, metrics, and limitations
└── requirements.txt
```

Run the lightweight tests with:

```bash
python -m unittest discover -s tests
```

## Upstream models

- [Salesforce BLIP VQA](https://huggingface.co/Salesforce/blip-vqa-base)
- [multilingual BERT](https://huggingface.co/google-bert/bert-base-multilingual-cased)
- [OpenAI CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32)

No ViVQA or MOCHEG samples are redistributed here. Obtain each dataset from its
original authors and follow its terms of use.

## Citation

Citation metadata is available in [`CITATION.cff`](CITATION.cff).

## License

A source-code license has not yet been selected. Add the authors' chosen
license before making this repository public. The upstream models and datasets
remain subject to their own licenses and terms.
