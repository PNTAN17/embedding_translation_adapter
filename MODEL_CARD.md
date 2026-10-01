# Model card

## Model description

The released models translate token embeddings from multilingual BERT into the
input space of a frozen English vision-language model. Both use a
source-conditioned dual-branch adapter. The ViVQA checkpoint targets BLIP's
768-dimensional text input space. The MOCHEG checkpoint targets the
512-dimensional representation consumed by its frozen stance detector.

## Intended use

- Research inference on Vietnamese visual question answering.
- Research inference on Vietnamese multimodal claim verification.
- Adapter training from prepared Vietnamese and English parallel text.
- Reproduction and inspection of the inference paths described in the paper.

The checkpoints are not intended for safety-critical or production fact
checking. A MOCHEG prediction is a model output, not a verified factual
judgment.

## Inputs and outputs

### ViVQA

Input is one RGB image and one Vietnamese question. Output is a short generated
answer string.

### MOCHEG

Input is one Vietnamese claim, one to five Vietnamese textual evidence items,
and zero to five evidence images. Output is a three-class prediction and its
softmax scores. The label order is `refuted`, `supported`, and `NEI`.

## Released checkpoint results

| Task | Split | Samples | Metric | Value |
|---|---:|---:|---|---:|
| ViVQA | test | 3,001 | exact match | 0.4585 |
| MOCHEG | test | 2,442 unique claims | micro F1 | 0.5025 |

Each row describes the single checkpoint in this repository. The checkpoints
were retained by validation performance. These values are not multi-seed
estimates.

## Dependencies

The adapter checkpoints do not contain the upstream backbone weights. Inference
loads the following frozen models separately.

- `bert-base-multilingual-cased`
- `Salesforce/blip-vqa-base` for ViVQA
- `openai/clip-vit-base-patch32` vision encoder for MOCHEG when images are used

## Limitations

- The MOCHEG Vietnamese claims and evidence used in the study were machine
  translated. Results do not establish performance on naturally authored
  Vietnamese fact-checking data.
- ViVQA exact match can mark semantically related answers as incorrect.
- Both tasks inherit biases and failure modes from their frozen upstream models
  and from translation-derived supervision.
- Inputs longer than 32 tokens for ViVQA or 77 tokens for MOCHEG are truncated.
- MOCHEG uses at most five textual and five image evidence items.
- Softmax scores are not calibrated confidence estimates.

## Artifact contents

`vivqa_adapter.pt` contains only the adapter state, architecture settings, and
run metadata. `mocheg_adapter.pt` additionally contains the frozen stance-head
state. Neither checkpoint contains data, credentials, machine-specific paths,
or the frozen upstream backbone weights.
