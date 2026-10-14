# Vietnamese Embedding Space Transfer

Reference code for **Teaching Frozen English Vision Language Models to Read
Vietnamese via Embedding Space Translation Distillation**.

ViVQA answers a Vietnamese question about an image using frozen BLIP. MOCHEG
predicts `refuted`, `supported`, or `NEI` from a Vietnamese claim, textual
evidence, and optional image evidence. The two adapters have separate weights.
ViVQA targets the complete BLIP input embedding module and passes its output
through `encoder_embeds` to the frozen text encoder. MOCHEG targets final CLIP
text states and passes its output directly to the frozen stance module.

Use this repository to prepare data, train adapters, run inference, and
evaluate saved predictions. Checkpoints are hosted on Hugging Face; see
[ARTIFACTS.md](ARTIFACTS.md) for the repository link.

## Installation

Use Python 3.10 or newer and install the dependencies in a virtual environment.
A CUDA GPU is recommended for training the full adapter.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader wordnet averaged_perceptron_tagger_eng punkt_tab
```

The NLTK resources are needed when evaluating ViVQA answers.
Transformers downloads frozen upstream weights on first use. Pass
`--local-files-only` after the models have been cached for offline execution.

## Inference

Download a compatible checkpoint from the repository linked in
[ARTIFACTS.md](ARTIFACTS.md), then pass its local file path to `--checkpoint`.

```bash
python infer_vivqa.py \
  --checkpoint /path/to/vivqa_adapter.pt \
  --image /path/to/image.jpg \
  --question "Có bao nhiêu người trong ảnh?"

python infer_mocheg.py \
  --checkpoint /path/to/mocheg_adapter.pt \
  --claim "Nội dung của tuyên bố cần kiểm chứng" \
  --evidence "Bằng chứng văn bản" \
  --image /path/to/evidence.jpg
```

ViVQA writes a JSON answer. MOCHEG writes a label, class probabilities, and the
number of evidence items used. Repeat `--evidence` and `--image` for more items.
MOCHEG uses up to five of each. Without images it uses the frozen text classifier.

## Data preparation and evaluation

`prepare_data.py` validates existing user supplied partitions and writes their
original IDs and file hashes. It removes training records that repeat a
normalized text in either held out partition. It does not invent partitions or
translate text. `evaluate.py` scores saved predictions with strict ID matching
and can also generate predictions from a supplied checkpoint.

```bash
python prepare_data.py --help
python evaluate.py --help
python train_vivqa.py --help
python train_mocheg.py --help
```

[TRAINING.md](TRAINING.md) gives input schemas and commands for data preparation,
training, and evaluation. For ViVQA, run `configs/vivqa_stage1.json` first, then
`configs/vivqa_stage2.json`. The Stage 2 configuration uses `init_checkpoint` to
load `outputs/vivqa_stage1_seed42.pt`. Change that field or pass
`--init-checkpoint /path/to/stage1.pt` when using a different checkpoint. Both
stages require the same seed and data, including 5,000 validation records.

## Translation generation

Install optional dependencies before using the translation utility.

```bash
pip install -r requirements-generation.txt
python generate_pairs.py --help
```

The utility translates user supplied Vietnamese records into English with
Gemini or OpenAI. It records generation settings and file hashes for new runs,
rejects incomplete provider output, and supports continuing unfinished files.
Use its dry run mode to inspect inputs before making paid requests.

## Repository contents

| Path | Purpose |
| --- | --- |
| `embedding_transfer` | Models, data validation, metrics, and provenance |
| `prepare_data.py` | Validate partitions and record exact supplied IDs |
| `evaluate.py` | Generate or score predictions |
| `generate_pairs.py` | Translate supplied text and record new generation settings |
| `train_vivqa.py` and `train_mocheg.py` | Adapter training and validation selection |
| `infer_vivqa.py` and `infer_mocheg.py` | Single example inference |
| `configs` | Executable reference settings and translation provenance template |
| `tests` | Tests that use small synthetic inputs and no pretrained weights |
| `ARTIFACTS.md` | Hugging Face checkpoint repository |
| `MODEL_CARD.md` | Model inputs, outputs, and checkpoint requirements |

Run the tests after installing the dependencies.

```bash
python -m unittest discover -s tests
```

## Upstream models and data

The code loads [mBERT](https://huggingface.co/google-bert/bert-base-multilingual-cased),
[BLIP VQA](https://huggingface.co/Salesforce/blip-vqa-base), and
[CLIP ViT B/32](https://huggingface.co/openai/clip-vit-base-patch32).
Obtain ViVQA, MOCHEG, and PhoMT from their authors and follow their terms.
No original, translated, or synthetic task samples are redistributed here.

## Citation and license

Citation metadata is in [CITATION.cff](CITATION.cff). A source code license is
not provided in this working copy. No new reuse permission is implied by these
changes. Upstream models and datasets retain their own licenses.
