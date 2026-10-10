# Model card

## Repository scope

This repository provides code for translating multilingual BERT token
embeddings into a selected interface of a frozen English vision-language
system. It does not publish model weights or claim a checkpoint-level result.
The ViVQA implementation targets BLIP's 768-dimensional text input embeddings.
The MOCHEG implementation targets the 512-dimensional final CLIP text states
consumed by a frozen stance detector. The CLIP text encoder is not run after
the adapter during Vietnamese inference.

## Intended use

- Research inference with a compatible user-supplied checkpoint.
- Adapter training from prepared Vietnamese and English parallel text.
- Inspection of the model interfaces described in the paper.

The code is not intended for safety-critical or production fact checking. A
MOCHEG prediction is a model output, not a verified factual judgment.

## Inputs and outputs

### ViVQA

Input is one RGB image and one Vietnamese question. Output is a short generated
answer string.

### MOCHEG

Input is one Vietnamese claim, one to five Vietnamese textual evidence items,
and zero to five evidence images. Output is a three-class prediction and its
softmax scores. The label order is `refuted`, `supported`, and `NEI`.

## Checkpoint compatibility

Inference requires an external checkpoint in the format validated by
`embedding_transfer/checkpoints.py`. A ViVQA payload contains the task name,
adapter configuration, and adapter state. A MOCHEG payload additionally
contains the frozen stance state. Users are responsible for verifying the
origin, selection protocol, and evaluation record of external weights.

## Dependencies

The repository does not bundle upstream backbone weights. Inference loads the
following frozen models separately.

- `bert-base-multilingual-cased`
- `Salesforce/blip-vqa-base` for ViVQA
- `openai/clip-vit-base-patch32` vision encoder for MOCHEG when images are used
