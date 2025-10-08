import argparse
import os, json, glob, math
from typing import Dict, List
from dataclasses import dataclass

import numpy as np
import torch
from datasets import Dataset, Audio, set_caching_enabled
from transformers import (
    AutoProcessor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("1", "true", "t", "yes", "y"):
        return True
    if value in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Cannot interpret '{value}' as boolean")

# ---------- SageMaker channels ----------
SM_CHANNEL_TRAIN = os.environ.get("SM_CHANNEL_TRAIN")        # e.g. /opt/ml/input/data/train
SM_CHANNEL_VALID = os.environ.get("SM_CHANNEL_VALIDATION")   # e.g. /opt/ml/input/data/validation

def find_manifest(base_dir: str, manifest_name: str) -> str:
    """Locate a manifest file for the given channel.

    The training job normally provides ``SM_CHANNEL_*`` directories that contain
    the manifest files.  When running the script locally (e.g., for debugging),
    those directories might not exist and the user may instead pass an absolute
    or relative path.  This helper tries a few sensible fallbacks before giving
    up so that the script works in both environments and the resulting error
    message is more actionable.
    """

    search_roots = []
    if manifest_name and os.path.isfile(manifest_name):
        return manifest_name

    if base_dir:
        if manifest_name:
            direct = os.path.join(base_dir, manifest_name)
            if os.path.isfile(direct):
                return direct
        if os.path.isdir(base_dir):
            search_roots.append(base_dir)

    # Also look in the current working tree.  This lets us run the script
    # locally with manifests that live next to the code checkout.
    search_roots.append(os.getcwd())

    candidates = []
    for root in search_roots:
        if manifest_name:
            candidates.extend(glob.glob(os.path.join(root, "**", manifest_name), recursive=True))
        if not candidates:
            candidates.extend(glob.glob(os.path.join(root, "**", "*.json*"), recursive=True))
        if candidates:
            break

    if not candidates:
        looked_in = ", ".join(sorted(set(search_roots))) or "<none>"
        raise FileNotFoundError(
            f"Could not find a manifest named '{manifest_name}' (searched: {looked_in})."
        )

    return candidates[0]

def s3_to_local(s3_uri: str, channel_dir: str) -> str:
    # s3://bucket/key -> /opt/ml/input/data/<channel>/key
    key = s3_uri.split("/", 3)[-1]
    if channel_dir:
        return os.path.join(channel_dir, key)
    # When we do not have a channel directory (e.g., local debugging), fall back
    # to the key itself so the caller can decide how to handle it.
    return key

def read_jsonl(path: str):
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def build_rows(manifest_path: str, channel_dir: str, label_col: str) -> List[Dict]:
    rows = []
    for ex in read_jsonl(manifest_path):
        s3u = ex.get("source-ref") or ex.get("source") or ex.get("wav") or ex.get("audio")
        if not s3u:
            continue
        local_path = s3_to_local(s3u, channel_dir)
        if not os.path.exists(local_path):
            # skip if object wasn't materialized (shouldn't happen with ManifestFile/File)
            continue
        lab = ex.get(label_col)
        rows.append({"audio": local_path, "label_text": lab})
    return rows

def make_label_maps(rows: List[Dict]):
    labels = sorted({r["label_text"] for r in rows})
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="facebook/mms-lid-126")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--train_manifest_name", type=str, required=True)
    ap.add_argument("--eval_manifest_name", type=str, required=True)
    ap.add_argument("--label_col", type=str, default="lang")
    ap.add_argument("--crop_seconds_train", type=float, default=2.0)
    ap.add_argument("--crop_seconds_eval", type=float, default=2.0)
    ap.add_argument("--num_proc", type=int, default=-1)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--fp16", type=str2bool, nargs="?", const=True, default=False)
    ap.add_argument("--output_dir", type=str, default="/opt/ml/model")
    args = ap.parse_args()

    # 1) Locate the manifests inside their channels
    train_manifest = find_manifest(SM_CHANNEL_TRAIN, args.train_manifest_name)
    eval_manifest  = find_manifest(SM_CHANNEL_VALID, args.eval_manifest_name)

    # 2) Build row lists from manifests (map s3 -> local)
    train_rows = build_rows(train_manifest, SM_CHANNEL_TRAIN, args.label_col)
    eval_rows  = build_rows(eval_manifest,  SM_CHANNEL_VALID, args.label_col)

    # 3) Label maps
    label2id, id2label = make_label_maps(train_rows + eval_rows)

    # 4) Hugging Face Datasets with on-demand decoding
    train_ds = Dataset.from_list(train_rows).cast_column("audio", Audio(sampling_rate=args.sr))
    eval_ds  = Dataset.from_list(eval_rows).cast_column("audio",  Audio(sampling_rate=args.sr))
    set_caching_enabled(True)

    # 5) Processor & featurization (pad/truncate to fixed seconds so we can avoid a custom collator)
    processor = AutoProcessor.from_pretrained(args.model_name)
    max_len_train = int(args.crop_seconds_train * args.sr)
    max_len_eval  = int(args.crop_seconds_eval * args.sr)

    def _prep(example, max_len):
        au = example["audio"]              # {"array": np.array, "sampling_rate": 16000}
        out = processor(
            au["array"],
            sampling_rate=au["sampling_rate"],
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_attention_mask=True,
        )
        example["input_values"]   = out["input_values"][0]
        example["attention_mask"] = out["attention_mask"][0]
        example["labels"]         = label2id[example["label_text"]]
        return example

    nproc = (os.cpu_count() if args.num_proc in (-1, 0) else args.num_proc)
    train_ds = train_ds.map(lambda e: _prep(e, max_len_train), num_proc=nproc, desc="featurize-train")
    eval_ds  = eval_ds.map(lambda e: _prep(e, max_len_eval),   num_proc=nproc, desc="featurize-eval")

    # Keep only model inputs
    train_ds = train_ds.remove_columns(["audio", "label_text"]).with_format("torch")
    eval_ds  = eval_ds.remove_columns(["audio", "label_text"]).with_format("torch")

    # 6) Model
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
        problem_type="single_label_classification",
    )

    # 7) Metrics
    def compute_metrics(pred):
        preds = np.argmax(pred.predictions, axis=1)
        acc = accuracy_score(pred.label_ids, preds)
        f1m = f1_score(pred.label_ids, preds, average="macro")
        return {"accuracy": acc, "f1_macro": f1m}

    # 8) Training args
    targs = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        fp16=args.fp16,
        dataloader_num_workers=max(2, (os.cpu_count() or 2)//2),
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=processor,  # lets Trainer save the processor with the model
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()
