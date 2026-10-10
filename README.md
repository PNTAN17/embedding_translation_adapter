# Vietnamese Embedding Space Transfer

Reference code for **Teaching Frozen English Vision Language Models to Read
Vietnamese via Embedding Space Translation Distillation**.

ViVQA answers a Vietnamese question about an image using frozen BLIP. MOCHEG
predicts `refuted`, `supported`, or `NEI` from a Vietnamese claim, textual
evidence, and optional image evidence. The two adapters have separate weights.
ViVQA targets the complete BLIP input embedding module and passes its output
through `encoder_embeds` to the frozen text encoder. MOCHEG targets final CLIP
text states and passes its output directly to the frozen stance module.

This working copy includes data checks, inference, evaluation, and adapter
training with either alignment loss or downstream validation selection. It
does not contain paper checkpoints, task data, translations, original paper
split IDs, or historical run logs. See [ARTIFACTS.md](ARTIFACTS.md) for the public
commit and the distinction between published files and these local changes.

## Installation

Use Python 3.10 or newer and install the dependencies in a virtual environment.
A CUDA GPU is recommended for training the full adapter.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader wordnet averaged_perceptron_tagger_eng punkt_tab
```

The NLTK resources are needed for ViVQA answer normalization. Missing resources
cause scoring to stop, so results never silently switch to another metric.
Transformers downloads frozen upstream weights on first use. Pass
`--local-files-only` after the models have been cached for offline execution.

## Inference

Supply a compatible checkpoint explicitly.

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

[TRAINING.md](TRAINING.md) gives schemas and runnable commands. The example
configurations enable validation Exact Match for ViVQA and the sum of
development micro F1 and macro F1 for MOCHEG. The command line defaults retain
alignment loss selection when task labels and images are unavailable. Neither
mode certifies reproduction of the paper without its original artifacts.

## Repository contents

| Path | Purpose |
| --- | --- |
| `embedding_transfer` | Models, data validation, metrics, and provenance |
| `prepare_data.py` | Validate partitions and record exact supplied IDs |
| `evaluate.py` | Generate or score predictions |
| `train_vivqa.py` and `train_mocheg.py` | Adapter training and validation selection |
| `infer_vivqa.py` and `infer_mocheg.py` | Single example inference |
| `configs` | Executable reference settings and translation provenance template |
| `tests` | Tests that use small synthetic inputs and no pretrained weights |
| `ARTIFACTS.md` | Public version and outstanding release artifacts |
| `MODEL_CARD.md` | Model scope and limitations |

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
not provided in this working copy or the verified public commit. No new reuse
permission is implied by this
working copy. Upstream models and datasets retain their own licenses.
