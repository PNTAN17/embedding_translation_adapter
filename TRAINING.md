# Training and evaluation

The entry points optimize only the adapter. The source embeddings, English
backbone, and MOCHEG stance module remain frozen. Validation labels are used
only for model selection. Neither trainer accepts a test file.

## Prepare existing partitions

Provide separate training, validation, and test JSONL files with original
record IDs. A parallel record has this schema.

```json
{"id": "original-question-id", "vi": "Câu hỏi", "en": "A question"}
```

Use the same schema for PhoMT pairs. MOCHEG uses parallel claims and evidence.

```json
{"id": "original-claim-id", "claim_vi": "Tuyên bố", "claim_en": "A claim", "evidence_vi": ["Bằng chứng"], "evidence_en": ["Evidence"]}
```

```bash
python prepare_data.py \
  --task parallel \
  --train /path/to/train.jsonl \
  --validation /path/to/validation.jsonl \
  --test /path/to/test.jsonl \
  --output-dir data/clean
```

Use `--task mocheg` for the second schema. JSONL and canonical column CSV are
accepted for parallel text. Use JSONL for MOCHEG evidence lists. The tool checks
that original IDs are unique and disjoint across the three inputs. It preserves
validation and test membership. It removes training records that repeat a
Vietnamese or English text from either held out partition, then removes
exact bilingual training duplicates. Distinct translations of one source are
preserved. MOCHEG training duplicates must match both the claim pair and the
complete evidence pair list, so shared training evidence alone does not remove
a claim. The conservative held out text check also includes evidence.
Repeated evidence can therefore exclude an otherwise distinct claim. Inspect
the recorded removal reasons before using the result.

`split_manifest.json` records input and output SHA256 hashes, all original IDs,
retained training IDs, and removal reasons. These are identifiers for the
supplied files. They are not a reconstruction of the paper splits or its fixed
validation subset. Supply that existing subset from the experiment machine.
The standalone trainers also check training versus validation overlap in each
language independently. They cannot check an unseen test file.

## ViVQA

For downstream validation, supply JSONL records with the same IDs and Vietnamese
questions as the validation pairs. The records may be a fixed subset of those
pairs. They must contain English references and image paths.

```json
{"id": "original-question-id", "question_vi": "Câu hỏi", "answer_en": "two", "image": "image.jpg"}
```

```bash
python train_vivqa.py \
  --config configs/vivqa_reference.json \
  --train-pairs data/clean/train.jsonl \
  --validation-pairs data/clean/validation.jsonl \
  --validation-task-data /path/to/fixed_validation_subset.jsonl \
  --validation-image-root /path/to/images \
  --seed 42 \
  --output outputs/vivqa_seed42.pt
```

Run separately with seeds 43 and 44 and distinct output names for repeated
runs. The configuration selects the earliest checkpoint with the highest
validation Exact Match. It uses patience ten and a maximum of thirty epochs.
It does not generate the paper's fixed subset or run new trials automatically.
Explicit command line values override JSON settings. Unknown settings fail.

ViVQA alignment targets the full BLIP embedding module. Inference therefore
uses `encoder_embeds` to avoid applying the embedding module twice. It uses the
source attention mask and greedy answer decoding. The former norm heuristic
could unmask padding because output LayerNorm gives padding vectors substantial
norms. These corrections apply to this reference implementation and do not
reconstruct historical predictions.

## MOCHEG

Task validation records contain the claim ID, Vietnamese claim and evidence,
one of the three labels, and an optional list of image paths.

```json
{"id": "original-claim-id", "claim_vi": "Tuyên bố", "evidence_vi": ["Bằng chứng"], "images": [], "label": "supported"}
```

```bash
python train_mocheg.py \
  --config configs/mocheg_reference.json \
  --train-pairs /path/to/mocheg_train.jsonl \
  --validation-pairs /path/to/mocheg_development.jsonl \
  --validation-task-data /path/to/mocheg_development_task.jsonl \
  --validation-image-root /path/to/images \
  --phomt-train /path/to/phomt_train.jsonl \
  --stance-checkpoint /path/to/compatible_mocheg_checkpoint.pt \
  --output outputs/mocheg_seed42.pt
```

PhoMT is optional. The stance checkpoint must contain an existing compatible
stance state. The script does not train or recreate that state. The example
configuration selects the earliest checkpoint maximizing development micro F1
plus macro F1. The plateau scheduler uses the same selection signal.
The chosen `max_evidence` limit is used for alignment, task validation, and
inference from the saved checkpoint. Validation evidence must match the paired
evidence subset for the same claim ID.

## Alignment only reference training

If task validation records are unavailable, omit `--config` and
`--validation-task-data`, or explicitly set `--selection-metric alignment_loss`.
Both trainers then select the smallest validation alignment loss. This option
provides a usable text alignment trainer but differs from the downstream
selection described in the paper. Use `--no-amp` for full precision and
`--local-files-only` for cached upstream models.

Every new checkpoint records the resolved scalar configuration, seed, input
hashes and IDs when provided, selected epoch, metric, package versions, code
hashes, and history through the selected epoch. The adjacent `.run.json` also
records every completed epoch, including epochs after the selected checkpoint.
The enclosing Git commit is informative only because files may have changed
locally. The per file hashes identify the code actually used. Paths to input
files are represented by basenames and hashes rather than machine specific
absolute paths. Exact upstream model revisions are not pinned by the examples.

## Evaluate saved predictions

ViVQA predictions use `id` and `prediction`. MOCHEG predictions use `id` and
`label`. The reference schemas are the task validation schemas above. Every
reference must have exactly one prediction with the same ID. Missing, extra,
and duplicate IDs cause an error.

```bash
python evaluate.py \
  --task vivqa \
  --references /path/to/test_references.jsonl \
  --predictions /path/to/test_predictions.jsonl \
  --output outputs/vivqa_metrics.json
```

To generate predictions, replace `--predictions` with the following options.

```text
--checkpoint /path/to/adapter.pt
--predictions-output predictions/test.jsonl
--image-root /path/to/images
```

ViVQA reports Exact Match and sample averaged token multiset F1 as percentages.
Normalization lowercases text, removes English articles and punctuation,
lemmatizes with NLTK POS tagging, then maps exact digit tokens from zero through
twenty to words. Install the resources documented in the README. MOCHEG reports
micro F1 and macro F1 over the fixed three label set. A class with no true or
predicted examples contributes zero to macro F1. Every report includes input
and prediction hashes. These scripts make new evaluations inspectable; they
do not supply missing predictions or establish the paper's reported scores.
