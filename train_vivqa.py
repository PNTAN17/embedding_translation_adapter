"""Train the ViVQA adapter."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch import nn
from transformers import (
    BertModel,
    BertTokenizer,
    BlipForQuestionAnswering,
    BlipProcessor,
)

from embedding_transfer.adapter import EmbeddingTransferAdapter
from embedding_transfer.checkpoints import load_payload
from embedding_transfer.provenance import (
    parse_config,
    run_provenance,
    save_history,
    validate_training_args,
)
from embedding_transfer.data import indexed_records, read_records
from embedding_transfer.metrics import predict_records, score_records
from embedding_transfer.training import (
    AlignmentLoss,
    choose_device,
    freeze,
    load_parallel_jsonl,
    make_loader,
    require_disjoint,
    save_payload,
    seed_everything,
)


SOURCE_MODEL = "bert-base-multilingual-cased"
TARGET_MODEL = "Salesforce/blip-vqa-base"


def validate_protocol(args, validation_records, task_records) -> None:
    """Validate training inputs and settings."""
    if args.gradient_accumulation < 1:
        raise ValueError("gradient accumulation must be positive")
    if args.training_stage == "stage1":
        if args.init_checkpoint is not None:
            raise ValueError("Stage 1 starts from random adapter weights")
        if args.early_stopping:
            raise ValueError("Stage 1 requires --no-early-stopping")
    if args.training_stage == "stage2" and args.init_checkpoint is None:
        raise ValueError("Stage 2 requires --init-checkpoint from Stage 1")
    outputs = (args.output, args.output.with_suffix(".run.json"))
    destinations = {
        path.expanduser().resolve()
        for output in outputs
        for path in (output, output.with_suffix(output.suffix + ".tmp"))
    }
    for name in (
        "train_pairs",
        "validation_pairs",
        "validation_task_data",
        "init_checkpoint",
        "config",
    ):
        path = getattr(args, name, None)
        if path is not None and path.expanduser().resolve() in destinations:
            raise ValueError(f"Checkpoint or history output would overwrite {name}")
    if args.expected_validation_count is not None:
        if args.expected_validation_count < 1:
            raise ValueError("expected validation count must be positive")
        if task_records is None or len(task_records) != args.expected_validation_count:
            raise ValueError(
                "Task validation count does not match --expected-validation-count"
            )
    if args.selection_metric == "exact_match" and task_records is None:
        raise ValueError("exact_match selection requires --validation-task-data")
    if args.validation_task_scope == "full" and task_records is None:
        raise ValueError("Full task validation requires --validation-task-data")
    if task_records is None:
        return
    known = indexed_records(validation_records)
    actual = indexed_records(task_records)
    if not actual.keys() <= known.keys():
        raise ValueError("Task validation IDs must be a subset of validation pair IDs")
    if args.validation_task_scope == "full" and actual.keys() != known.keys():
        raise ValueError("Full task validation must cover every validation pair ID")
    for identifier, row in actual.items():
        for key in ("question_vi", "answer_en", "image"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Task validation requires a nonempty {key!r}")
        if row["question_vi"].strip() != known[identifier]["vi"].strip():
            raise ValueError("Task validation questions must match their bilingual IDs")


def initialize_adapter(adapter, config, args, provenance) -> dict | None:
    """Initialize adapter weights from a checkpoint."""
    if args.init_checkpoint is None:
        return None
    payload = load_payload(args.init_checkpoint, "vivqa")
    if payload["adapter_config"] != config:
        raise ValueError("Initialization checkpoint adapter configuration differs")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Initialization checkpoint requires model metadata")
    for key, expected in (
        ("source_model", SOURCE_MODEL),
        ("target_model", TARGET_MODEL),
    ):
        if metadata.get(key) != expected:
            raise ValueError(f"Initialization checkpoint has an incompatible {key}")
    if args.training_stage == "stage2":
        if metadata.get("training_stage") != "stage1":
            raise ValueError("Stage 2 must initialize from a Stage 1 checkpoint")
        if metadata.get("seed") != args.seed:
            raise ValueError("Stage 2 must use the same seed as Stage 1")
        if (
            metadata.get("selection_split") != "validation"
            or metadata.get("selection_metric") != args.selection_metric
            or not isinstance(metadata.get("selected_epoch"), int)
            or metadata["selected_epoch"] < 1
        ):
            raise ValueError(
                "Stage 1 must have a compatible validation-selected checkpoint"
            )
        previous_provenance = metadata.get("provenance")
        if not isinstance(previous_provenance, dict) or not isinstance(
            previous_provenance.get("inputs"), dict
        ):
            raise ValueError("Stage 1 requires input-artifact provenance")
        previous_inputs = previous_provenance["inputs"]
        for name in ("train_pairs", "validation_pairs", "validation_task_data"):
            previous = previous_inputs.get(name, {}).get("sha256")
            current = provenance["inputs"].get(name, {}).get("sha256")
            if previous != current or (
                name != "validation_task_data" and current is None
            ):
                raise ValueError(f"Stage 2 must retain the Stage 1 {name} data")
    for name, value in payload["adapter_state_dict"].items():
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError(
                f"Initialization contains invalid adapter weights at {name!r}"
            )
    adapter.load_state_dict(payload["adapter_state_dict"], strict=True)
    return {
        "mode": "adapter_weights_only",
        "checkpoint": provenance["inputs"]["init_checkpoint"],
        "source_stage": metadata.get("training_stage"),
        "source_seed": metadata.get("seed"),
        "source_selected_epoch": metadata.get("selected_epoch"),
        "optimizer_and_scheduler": "new",
    }


def tokenize(tokenizer, texts: list[str], length: int, device: torch.device):
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=length,
        return_tensors="pt",
    ).to(device)


def batch_loss(
    records: list[dict],
    adapter: nn.Module,
    source_embeddings: nn.Module,
    target_embeddings: nn.Module,
    source_tokenizer,
    target_tokenizer,
    criterion: nn.Module,
    length: int,
    device: torch.device,
    amp: bool,
) -> torch.Tensor:
    source = tokenize(
        source_tokenizer, [item["vi"] for item in records], length, device
    )
    target = tokenize(
        target_tokenizer, [item["en"] for item in records], length, device
    )
    with (
        torch.no_grad(),
        torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp),
    ):
        source_values = source_embeddings(input_ids=source.input_ids)
        target_values = target_embeddings(input_ids=target.input_ids)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        prediction = adapter(source_values, source.attention_mask)
        return criterion(
            prediction,
            target_values,
            source.attention_mask,
            target.attention_mask,
        )


def train_epoch(
    loader,
    adapter,
    source_embeddings,
    target_embeddings,
    source_tokenizer,
    target_tokenizer,
    criterion,
    optimizer,
    scheduler,
    scaler,
    args,
    device,
) -> float:
    adapter.train()
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    samples = 0
    for batch_index, records in enumerate(loader):
        group_offset = batch_index % args.gradient_accumulation
        if group_offset == 0:
            group_size = min(args.gradient_accumulation, len(loader) - batch_index)
            group_examples = min(
                group_size * args.batch_size,
                len(loader.dataset) - batch_index * args.batch_size,
            )
        loss = batch_loss(
            records,
            adapter,
            source_embeddings,
            target_embeddings,
            source_tokenizer,
            target_tokenizer,
            criterion,
            args.max_length,
            device,
            args.amp and device.type == "cuda",
        )
        scaler.scale(loss * len(records) / group_examples).backward()
        total += loss.detach().item() * len(records)
        samples += len(records)
        end_of_group = group_offset + 1 == group_size
        if end_of_group:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
    return total / samples


@torch.inference_mode()
def validate(
    loader,
    adapter,
    source_embeddings,
    target_embeddings,
    source_tokenizer,
    target_tokenizer,
    criterion,
    args,
    device,
) -> float:
    adapter.eval()
    total = 0.0
    samples = 0
    for records in loader:
        loss = batch_loss(
            records,
            adapter,
            source_embeddings,
            target_embeddings,
            source_tokenizer,
            target_tokenizer,
            criterion,
            args.max_length,
            device,
            args.amp and device.type == "cuda",
        )
        total += loss.item() * len(records)
        samples += len(records)
    return total / samples


def main() -> None:
    args = parse_config(build_parser())
    validate_training_args(args)
    seed_everything(args.seed)
    device = choose_device(args.device)

    train_records = load_parallel_jsonl(args.train_pairs)
    validation_records = load_parallel_jsonl(args.validation_pairs)
    require_disjoint(train_records, validation_records)
    task_records = (
        read_records(args.validation_task_data) if args.validation_task_data else None
    )
    validate_protocol(args, validation_records, task_records)
    provenance = run_provenance(
        args,
        {
            "train_pairs": args.train_pairs,
            "validation_pairs": args.validation_pairs,
            "validation_task_data": args.validation_task_data,
            "init_checkpoint": args.init_checkpoint,
        },
    )
    train_loader = make_loader(train_records, args.batch_size, True, args.seed)
    validation_loader = make_loader(
        validation_records, args.batch_size, False, args.seed
    )

    config = {
        "input_embedding_dim": 768,
        "output_embedding_dim": 768,
        "hidden_dim": args.hidden_dim,
        "num_encoder_layers": args.encoder_layers,
        "num_decoder_layers": args.decoder_layers,
        "num_heads": args.heads,
        "dropout": args.dropout,
        "max_seq_length": args.max_length,
    }
    adapter = EmbeddingTransferAdapter(**config).to(device)
    initialization = initialize_adapter(adapter, config, args, provenance)
    provenance["initialization"] = initialization

    source_tokenizer = BertTokenizer.from_pretrained(
        SOURCE_MODEL, local_files_only=args.local_files_only
    )
    target_processor = BlipProcessor.from_pretrained(
        TARGET_MODEL, local_files_only=args.local_files_only, use_fast=False
    )
    target_tokenizer = target_processor.tokenizer
    source_model = BertModel.from_pretrained(
        SOURCE_MODEL, local_files_only=args.local_files_only
    )
    target_model = BlipForQuestionAnswering.from_pretrained(
        TARGET_MODEL, local_files_only=args.local_files_only
    )
    source_embeddings = freeze(source_model.embeddings.to(device))
    target_embeddings = freeze(target_model.text_encoder.embeddings.to(device))
    del source_model

    predictor = None
    if task_records is not None:
        from embedding_transfer.vivqa import ViVQAPredictor

        predictor = ViVQAPredictor.from_modules(
            adapter,
            freeze(target_model.to(device)),
            source_embeddings,
            source_tokenizer,
            target_processor,
            device,
        )
    else:
        del target_model, target_processor
    criterion = AlignmentLoss().to(device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=updates_per_epoch * args.epochs,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=25,
        final_div_factor=1000,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    best = float("inf")
    selected_epoch = None
    stale_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_value = train_epoch(
            train_loader,
            adapter,
            source_embeddings,
            target_embeddings,
            source_tokenizer,
            target_tokenizer,
            criterion,
            optimizer,
            scheduler,
            scaler,
            args,
            device,
        )
        validation_value = validate(
            validation_loader,
            adapter,
            source_embeddings,
            target_embeddings,
            source_tokenizer,
            target_tokenizer,
            criterion,
            args,
            device,
        )
        if not math.isfinite(train_value) or not math.isfinite(validation_value):
            raise RuntimeError("Nonfinite alignment loss; no checkpoint was selected")
        task_metrics = (
            score_records(
                task_records,
                predict_records(
                    task_records, predictor, "vivqa", args.validation_image_root
                ),
                "vivqa",
            )
            if predictor
            else {}
        )
        selection_value = (
            -task_metrics["exact_match"]
            if args.selection_metric == "exact_match"
            else validation_value
        )
        history.append(
            {
                "epoch": epoch,
                "train_alignment_loss": train_value,
                "validation_alignment_loss": validation_value,
                "validation_task_metrics": task_metrics,
            }
        )
        print(
            f"epoch={epoch} train_alignment_loss={train_value:.6f} "
            f"validation_alignment_loss={validation_value:.6f}"
        )
        if selection_value < best:
            best = selection_value
            selected_epoch = epoch
            stale_epochs = 0
            save_payload(
                {
                    "format_version": 1,
                    "task": "vivqa",
                    "adapter_config": config,
                    "adapter_state_dict": adapter.state_dict(),
                    "metadata": {
                        "seed": args.seed,
                        "training_stage": args.training_stage,
                        "initialization": initialization,
                        "selected_epoch": epoch,
                        "selection_split": "validation",
                        "selection_metric": args.selection_metric,
                        "validation_alignment_loss": validation_value,
                        "validation_task_metrics": task_metrics,
                        "provenance": provenance,
                        "source_model": SOURCE_MODEL,
                        "target_model": TARGET_MODEL,
                        "train_pairs": len(train_records),
                        "validation_pairs": len(validation_records),
                        "validation_task_count": len(task_records)
                        if task_records is not None
                        else 0,
                        "validation_task_scope": args.validation_task_scope,
                        "history": history,
                    },
                },
                args.output,
            )
        else:
            stale_epochs += 1
        save_history(args.output, provenance, history, selected_epoch)
        if args.early_stopping and stale_epochs:
            if stale_epochs >= args.patience:
                print(f"stopped after {args.patience} epochs without improvement")
                break
    print(
        f"saved={Path(args.output).expanduser().resolve()} selected_epoch={selected_epoch} metric={args.selection_metric}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-pairs", type=Path, required=True)
    parser.add_argument("--validation-pairs", type=Path, required=True)
    parser.add_argument("--validation-task-data", type=Path)
    parser.add_argument("--validation-image-root", type=Path, default=Path("."))
    parser.add_argument(
        "--validation-task-scope",
        choices=("subset", "full"),
        default="subset",
        help="Task validation scope.",
    )
    parser.add_argument(
        "--expected-validation-count",
        type=int,
        help="Expected number of task validation records.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("alignment_loss", "exact_match"),
        default="alignment_loss",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/vivqa_adapter.pt"))
    parser.add_argument(
        "--training-stage",
        choices=("standalone", "stage1", "stage2"),
        default="standalone",
        help="Training stage.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="Checkpoint for initializing adapter weights.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--early-stopping", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=1280)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


if __name__ == "__main__":
    main()
