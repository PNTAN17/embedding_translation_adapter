"""Train the MOCHEG embedding adapter from prepared bilingual JSONL files."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch import nn
from transformers import BertModel, BertTokenizer, CLIPModel, CLIPTokenizer

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
    load_mocheg_jsonl,
    load_parallel_jsonl,
    make_loader,
    make_plateau_scheduler,
    require_disjoint,
    save_payload,
    seed_everything,
)


SOURCE_MODEL = "bert-base-multilingual-cased"
TARGET_MODEL = "openai/clip-vit-base-patch32"


def tokenize(tokenizer, texts: list[str], length: int, device: torch.device):
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=length,
        return_tensors="pt",
    ).to(device)


def aligned_tensors(
    pairs: list[dict],
    adapter: nn.Module,
    source_embeddings: nn.Module,
    target_model: nn.Module,
    source_tokenizer,
    target_tokenizer,
    length: int,
    device: torch.device,
    amp: bool,
):
    source = tokenize(source_tokenizer, [item["vi"] for item in pairs], length, device)
    target = tokenize(target_tokenizer, [item["en"] for item in pairs], length, device)
    with (
        torch.no_grad(),
        torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp),
    ):
        source_values = source_embeddings(source.input_ids)
        target_values = target_model(
            input_ids=target.input_ids, attention_mask=target.attention_mask
        ).last_hidden_state
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        prediction = adapter(source_values, source.attention_mask)
    return prediction, target_values, source.attention_mask, target.attention_mask


def batch_loss(
    records,
    adapter,
    source_embeddings,
    target_model,
    source_tokenizer,
    target_tokenizer,
    criterion,
    args,
    device,
):
    amp = args.amp and device.type == "cuda"
    primary = [dict(vi=item["vi"], en=item["en"]) for item in records]
    tensors = aligned_tensors(
        primary,
        adapter,
        source_embeddings,
        target_model,
        source_tokenizer,
        target_tokenizer,
        args.max_length,
        device,
        amp,
    )
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        primary_loss = criterion(*tensors)

    groups = [item.get("evidence", []) for item in records]
    evidence = [pair for group in groups for pair in group]
    evidence_loss = primary_loss.new_zeros(())
    if evidence:
        prediction, target, source_mask, target_mask = aligned_tensors(
            evidence,
            adapter,
            source_embeddings,
            target_model,
            source_tokenizer,
            target_tokenizer,
            args.max_length,
            device,
            amp,
        )
        losses = []
        offset = 0
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            for group in groups:
                count = len(group)
                if count:
                    section = slice(offset, offset + count)
                    losses.append(
                        criterion(
                            prediction[section],
                            target[section],
                            source_mask[section],
                            target_mask[section],
                        )
                    )
                    offset += count
            evidence_loss = torch.stack(losses).mean()
    return args.claim_weight * primary_loss + args.evidence_weight * evidence_loss


def train_epoch(
    loader,
    adapter,
    source_embeddings,
    target_model,
    source_tokenizer,
    target_tokenizer,
    criterion,
    optimizer,
    scaler,
    args,
    device,
) -> float:
    adapter.train()
    total = 0.0
    samples = 0
    for records in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = batch_loss(
            records,
            adapter,
            source_embeddings,
            target_model,
            source_tokenizer,
            target_tokenizer,
            criterion,
            args,
            device,
        )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total += loss.detach().item() * len(records)
        samples += len(records)
    return total / samples


@torch.inference_mode()
def validate(
    loader,
    adapter,
    source_embeddings,
    target_model,
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
            target_model,
            source_tokenizer,
            target_tokenizer,
            criterion,
            args,
            device,
        )
        total += loss.item() * len(records)
        samples += len(records)
    return total / samples


def main() -> None:
    args = parse_config(build_parser())
    validate_training_args(args)
    if args.patience < 1:
        raise ValueError("patience must be positive")
    seed_everything(args.seed)
    device = choose_device(args.device)

    mocheg_train = load_mocheg_jsonl(args.train_pairs, args.max_evidence)
    validation_records = load_mocheg_jsonl(args.validation_pairs, args.max_evidence)
    phomt_records = load_parallel_jsonl(args.phomt_train) if args.phomt_train else []
    train_records = mocheg_train + phomt_records
    require_disjoint(train_records, validation_records)
    task_records = (
        read_records(args.validation_task_data) if args.validation_task_data else None
    )
    if args.selection_metric == "micro_macro_f1_sum" and task_records is None:
        raise ValueError("micro_macro_f1_sum selection requires --validation-task-data")
    if task_records is not None:
        known = indexed_records(validation_records)
        if not indexed_records(task_records).keys() <= known.keys():
            raise ValueError(
                "Task validation IDs must be a subset of validation pair IDs"
            )
        for row in task_records:
            if row["claim_vi"].strip() != known[str(row["id"])]["vi"].strip():
                raise ValueError(
                    "Task validation claims must match their bilingual IDs"
                )
            if [item.strip() for item in row["evidence_vi"][: args.max_evidence]] != [
                pair["vi"] for pair in known[str(row["id"])]["evidence"]
            ]:
                raise ValueError(
                    "Task validation evidence must match the paired evidence subset"
                )
    provenance = run_provenance(
        args,
        {
            "train_pairs": args.train_pairs,
            "validation_pairs": args.validation_pairs,
            "phomt_train": args.phomt_train,
            "stance_checkpoint": args.stance_checkpoint,
            "validation_task_data": args.validation_task_data,
        },
    )
    train_loader = make_loader(train_records, args.batch_size, True, args.seed)
    validation_loader = make_loader(
        validation_records, args.batch_size, False, args.seed
    )

    source_tokenizer = BertTokenizer.from_pretrained(
        SOURCE_MODEL, local_files_only=args.local_files_only
    )
    target_tokenizer = CLIPTokenizer.from_pretrained(
        TARGET_MODEL, local_files_only=args.local_files_only
    )
    source_model = BertModel.from_pretrained(
        SOURCE_MODEL, local_files_only=args.local_files_only
    )
    clip_model = CLIPModel.from_pretrained(
        TARGET_MODEL, local_files_only=args.local_files_only
    )
    source_embeddings = freeze(source_model.embeddings.word_embeddings.to(device))
    target_model = freeze(clip_model.text_model.to(device))
    del source_model, clip_model

    reference_payload = load_payload(args.stance_checkpoint, "mocheg")
    stance_state = reference_payload["stance_state_dict"]
    del reference_payload
    config = {
        "input_embedding_dim": 768,
        "output_embedding_dim": 512,
        "hidden_dim": args.hidden_dim,
        "num_encoder_layers": args.encoder_layers,
        "num_decoder_layers": args.decoder_layers,
        "num_heads": args.heads,
        "dropout": args.dropout,
        "max_seq_length": args.max_length,
    }
    adapter = EmbeddingTransferAdapter(**config).to(device)
    predictor = None
    if task_records is not None:
        from embedding_transfer.mocheg import MOCHEGPredictor

        predictor = MOCHEGPredictor.from_modules(
            adapter,
            stance_state,
            source_embeddings,
            source_tokenizer,
            device,
            args.local_files_only,
            args.max_evidence,
        )
    criterion = AlignmentLoss().to(device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = make_plateau_scheduler(optimizer)
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
            target_model,
            source_tokenizer,
            target_tokenizer,
            criterion,
            optimizer,
            scaler,
            args,
            device,
        )
        validation_value = validate(
            validation_loader,
            adapter,
            source_embeddings,
            target_model,
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
                    task_records, predictor, "mocheg", args.validation_image_root
                ),
                "mocheg",
            )
            if predictor
            else {}
        )
        selection_value = (
            -(task_metrics["micro_f1"] + task_metrics["macro_f1"])
            if args.selection_metric == "micro_macro_f1_sum"
            else validation_value
        )
        scheduler.step(selection_value)
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
                    "task": "mocheg",
                    "adapter_config": config,
                    "adapter_state_dict": adapter.state_dict(),
                    "stance_state_dict": stance_state,
                    "metadata": {
                        "seed": args.seed,
                        "max_evidence": args.max_evidence,
                        "selected_epoch": epoch,
                        "selection_split": "validation",
                        "selection_metric": args.selection_metric,
                        "validation_alignment_loss": validation_value,
                        "validation_task_metrics": task_metrics,
                        "provenance": provenance,
                        "source_model": SOURCE_MODEL,
                        "target_model": TARGET_MODEL,
                        "mocheg_train_claims": len(mocheg_train),
                        "phomt_train_pairs": len(phomt_records),
                        "validation_claims": len(validation_records),
                        "history": history,
                    },
                },
                args.output,
            )
        else:
            stale_epochs += 1
        save_history(args.output, provenance, history, selected_epoch)
        if stale_epochs:
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
        "--selection-metric",
        choices=("alignment_loss", "micro_macro_f1_sum"),
        default="alignment_loss",
    )
    parser.add_argument("--phomt-train", type=Path)
    parser.add_argument(
        "--stance-checkpoint",
        type=Path,
        required=True,
        help="Compatible checkpoint containing the frozen stance state.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/mocheg_adapter.pt")
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--claim-weight", type=float, default=1.0)
    parser.add_argument("--evidence-weight", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=1280)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=77)
    parser.add_argument("--max-evidence", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


if __name__ == "__main__":
    main()
