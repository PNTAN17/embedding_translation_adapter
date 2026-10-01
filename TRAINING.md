# Training

The two entry points train only the embedding adapter. The multilingual BERT
source module and the English target module remain frozen. MOCHEG also keeps
the released stance head frozen.

The scripts consume prepared JSONL files. They do not create dataset splits or
download task data. Prepare training and held-out validation files separately
and keep the latter out of all optimization data. Before training starts, each
script compares normalized bilingual pairs and stops if the two inputs share a
pair.

## ViVQA adapter

Each line in both ViVQA files contains one Vietnamese and English question.

```json
{"vi": "Có bao nhiêu người trong ảnh?", "en": "How many people are in the image?"}
```

Run the trainer with:

```bash
python train_vivqa.py \
  --train-pairs /path/to/vivqa_train_pairs.jsonl \
  --validation-pairs /path/to/vivqa_validation_pairs.jsonl \
  --output outputs/vivqa_adapter.pt
```

The defaults match the architecture and optimization settings reported in the
paper. They use a hidden size of 1280, four encoder layers, two decoder layers,
16 attention heads, a maximum length of 32, AdamW, and a OneCycle schedule with
a peak learning rate of `1e-4`.

## MOCHEG adapter

Each MOCHEG line contains a translated claim and parallel evidence lists. Only
the first five evidence pairs are used by default.

```json
{"claim_vi": "Một tuyên bố tiếng Việt", "claim_en": "An English claim", "evidence_vi": ["Bằng chứng"], "evidence_en": ["Evidence"]}
```

PhoMT is optional. Its JSONL format is the same two-field format used by the
ViVQA trainer.

```bash
python train_mocheg.py \
  --train-pairs /path/to/mocheg_train_pairs.jsonl \
  --validation-pairs /path/to/mocheg_validation_pairs.jsonl \
  --phomt-train /path/to/phomt_train_pairs.jsonl \
  --output outputs/mocheg_adapter.pt
```

The MOCHEG defaults use the same adapter architecture with a 512-dimensional
output, a maximum length of 77, AdamW with learning rate `1e-5`, and a plateau
scheduler. Claim and evidence alignment have equal weight. The stance state is
copied from `checkpoints/mocheg_adapter.pt` so the new checkpoint can be loaded
by `infer_mocheg.py`. It is never added to the optimizer.

## Model selection

Both scripts select checkpoints only by held-out validation alignment loss.
Their command line interfaces have no evaluation-data argument. The saved
metadata records the selected epoch, seed, validation loss, model identifiers,
sample counts, and loss history. It does not store local data paths.

The released checkpoints under `checkpoints/` are the retained paper runs.
New training runs are compatible with the inference commands, but their exact
scores can vary with the prepared parallel data, software stack, and hardware.

Use `--no-amp` when full precision is required. Use `--local-files-only` after
the three upstream models have been cached locally.
