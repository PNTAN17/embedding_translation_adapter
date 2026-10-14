# Training and evaluation

The entry points optimize only the adapter. The source embeddings, English
backbone, and MOCHEG stance module remain frozen. Validation labels are used
only for model selection. Neither trainer accepts a test file.

Run the commands below from the repository root. Relative paths in JSON
configurations are resolved from the current working directory. Explicit
command line options override JSON settings.

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
retained training IDs, and removal reasons.
The standalone trainers also check training versus validation overlap in each
language independently. They cannot check an unseen test file.

## ViVQA

For downstream validation, supply JSONL records with the same IDs and Vietnamese
questions as the validation pairs. Generic training permits a declared subset,
whereas the stage configurations require the complete supplied validation
partition of 5,000 records. Each record must contain an English reference and
an image path.

```json
{"id": "original-question-id", "question_vi": "Câu hỏi", "answer_en": "two", "image": "image.jpg"}
```

```bash
python train_vivqa.py \
  --config configs/vivqa_reference.json \
  --train-pairs data/clean/train.jsonl \
  --validation-pairs data/clean/validation.jsonl \
  --validation-task-data /path/to/validation_task.jsonl \
  --validation-image-root /path/to/images \
  --seed 42 \
  --output outputs/vivqa_seed42.pt
```

This configuration runs a single training stage with a maximum of thirty
epochs and early stopping patience of ten. Use distinct output names when
running with different seeds. Unknown JSON settings cause an error.

### The two stage ViVQA schedule

Stage 1 begins with random adapter weights and runs thirty epochs with
`early_stopping` set to `false` and maximum learning rate 0.0002. Stage 2 loads
the selected Stage 1 adapter checkpoint, starts a fresh optimizer and scheduler,
and uses maximum learning rate 0.0001. Stage 2 enables early stopping with
patience ten and a maximum of thirty epochs.

Set the checkpoint path in `configs/vivqa_stage2.json` using `init_checkpoint`.
The supplied configuration already contains this entry:

```json
{
  "init_checkpoint": "outputs/vivqa_stage1_seed42.pt"
}
```

This path matches the `output` in `configs/vivqa_stage1.json`. Run both stages
in order:

```bash
python train_vivqa.py \
  --config configs/vivqa_stage1.json \
  --train-pairs data/clean/train.jsonl \
  --validation-pairs data/clean/validation.jsonl \
  --validation-task-data /path/to/validation_task.jsonl \
  --validation-image-root /path/to/images \
  --seed 42 \
  --output outputs/vivqa_stage1_seed42.pt

python train_vivqa.py \
  --config configs/vivqa_stage2.json \
  --train-pairs data/clean/train.jsonl \
  --validation-pairs data/clean/validation.jsonl \
  --validation-task-data /path/to/validation_task.jsonl \
  --validation-image-root /path/to/images \
  --seed 42 \
  --output outputs/vivqa_stage2_seed42.pt
```

To use a different checkpoint, edit `init_checkpoint` or add
`--init-checkpoint /path/to/stage1.pt` to the Stage 2 command. Use a local `.pt`
file path; download hosted checkpoints before running the trainer.

For another seed, update `--seed` and `--output` for both stages and point
Stage 2 at that seed's Stage 1 output. For example, after training Stage 1 with
`--seed 43 --output outputs/vivqa_stage1_seed43.pt`, add these options to the
Stage 2 command:

```text
--seed 43
--init-checkpoint outputs/vivqa_stage1_seed43.pt
--output outputs/vivqa_stage2_seed43.pt
```

Stage 2 requires matching adapter settings, source and target model identifiers,
seed, selection settings, and the same three input data files (`train_pairs`,
`validation_pairs`, and `validation_task_data`). Keep a separate output path
for each stage. Initialization loads adapter weights only; optimizer, scheduler,
and epoch counters start fresh.

The stage files set `validation_task_scope` to `full` and
`expected_validation_count` to 5000. Supply task records for every validation
pair ID. If your complete validation partition has a different size, update
`expected_validation_count` in both configurations before running Stage 1.

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
stance state. The script does not train or recreate that state.
The chosen `max_evidence` limit is used for alignment, task validation, and
inference from the saved checkpoint. Task selection requires every supplied
development pair ID. Validation evidence must match the paired evidence subset
for the same claim ID. Training drops a bilingual evidence pair only when both
sides are empty, then takes the first nonempty pairs up to the evidence limit.
A pair with text on just one side fails validation. Data preparation remains
strict and expects nonempty evidence pairs in its input files.

## Alignment only reference training

If task validation records are unavailable, omit `--config` and
`--validation-task-data`. Both trainers default to
`--selection-metric alignment_loss` for this setup. Use `--no-amp` for full
precision and `--local-files-only` for cached upstream models.

The `.pt` output stores adapter weights, configuration, seed, input hashes,
selected epoch, and run metadata. The adjacent `.run.json` stores the history
of every completed epoch.

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

Install the NLTK resources listed in [README.md](README.md) before evaluating
ViVQA. For MOCHEG, pass `--task mocheg` with MOCHEG reference and prediction
files. The evaluator writes its JSON report to the path passed to `--output`.
