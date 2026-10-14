# Model card

## Repository scope

This repository provides code for translating multilingual BERT token
embeddings into a selected interface of a frozen English vision-language
system. Download checkpoints from the Hugging Face repository linked in
[ARTIFACTS.md](ARTIFACTS.md).
The ViVQA implementation targets BLIP's 768-dimensional text input embeddings.
The MOCHEG implementation targets the 512-dimensional final CLIP text states
consumed by a frozen stance detector. The CLIP text encoder is not run after
the adapter during Vietnamese inference.

## Intended use

- Inference with a compatible checkpoint downloaded to a local file.
- Adapter training from prepared Vietnamese and English parallel text.

See [README.md](README.md) for installation and inference commands, and
[TRAINING.md](TRAINING.md) for data schemas and training commands.

## Inputs and outputs

### ViVQA

Input is one RGB image and one Vietnamese question. Output is a short generated
answer string.

### MOCHEG

Input is one Vietnamese claim, one to five Vietnamese textual evidence items,
and zero to five evidence images. Output is a three-class prediction and its
softmax scores. The label order is `refuted`, `supported`, and `NEI`.

## Checkpoint compatibility

Pass the local checkpoint path to `--checkpoint` when running inference.
The format is validated by `embedding_transfer/checkpoints.py`. A ViVQA payload
contains the task name, adapter configuration, and adapter state. A MOCHEG
payload additionally contains the frozen stance state.

For ViVQA Stage 2 training, set `init_checkpoint` in
`configs/vivqa_stage2.json` or pass `--init-checkpoint /path/to/stage1.pt`.
Use a Stage 1 checkpoint trained with the same seed and input data.

## Dependencies

The repository does not bundle upstream backbone weights. Inference loads the
following frozen models separately.

- `bert-base-multilingual-cased`
- `Salesforce/blip-vqa-base` for ViVQA
- `openai/clip-vit-base-patch32` vision encoder for MOCHEG when images are used
