"""Split FiscaAI/synth-ehr-icd10cm-prompt into fixed train/val/test JSONL files.

Writes:
    splits/synth/train.jsonl  (80%)
    splits/synth/val.jsonl    (10%)
    splits/synth/test.jsonl   (10%)
    splits/synth/indices.json (row indices per split, for verification)

Training stays unchanged: point it at the train file with
    --data_source jsonl --data_path splits/synth/train.jsonl
The script's internal 90/10 split then acts as an additional val split drawn
only from train data. val.jsonl and test.jsonl are never seen in training;
use test.jsonl for the final method comparison.
"""
import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset, load_from_disk
import pandas as pd


def parse_codes(item):
    c = item.get("codes")
    if isinstance(c, str):
        c = c.strip().lstrip("[").rstrip("]")
        item["codes"] = [x.strip(" '\"") for x in c.split(",") if x.strip(" '\"")]
    elif not isinstance(c, list):
        item["codes"] = []
    return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synth_dataset", type=str, default="FiscaAI/synth-ehr-icd10cm-prompt")
    parser.add_argument("--synth_cache_dir", type=str, default="FiscaAI___synth-ehr-icd10cm-prompt")
    parser.add_argument("--output_dir", type=str, default="./splits/synth")
    parser.add_argument("--seed", type=int, default=42,
                        help="Must match training --seed so ordering matches")
    parser.add_argument("--val_frac", type=float, default=0.10)
    parser.add_argument("--test_frac", type=float, default=0.10)
    args = parser.parse_args()

    try:
        dataset = load_dataset(args.synth_dataset, cache_dir=args.synth_cache_dir)
        print(f"Loaded from HF hub: {args.synth_dataset}")
    except Exception as e:
        print(f"load_dataset failed ({e}), trying load_from_disk...")
        dataset = load_from_disk(args.synth_cache_dir)

    split = "train" if "train" in dataset else list(dataset.keys())[0]
    all_data = [parse_codes(dict(item)) for item in dataset[split]]
    print(f"Total rows: {len(all_data)}")

    random.seed(args.seed)
    indices = list(range(len(all_data)))
    random.shuffle(indices)

    n_test = int(len(indices) * args.test_frac)
    n_val = int(len(indices) * args.val_frac)
    test_idx = sorted(indices[:n_test])
    val_idx = sorted(indices[n_test:n_test + n_val])
    train_idx = sorted(indices[n_test + n_val:])

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def dump(name, idx):
        path = out / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for i in idx:
                f.write(json.dumps(all_data[i], ensure_ascii=False) + "\n")
        print(f"{name}: {len(idx)} rows -> {path}")

    dump("train", train_idx)
    dump("val", val_idx)
    dump("test", test_idx)

    with open(out / "indices.json", "w") as f:
        json.dump({
            "seed": args.seed,
            "n_total": len(all_data),
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        }, f)
    print(f"Indices saved -> {out / 'indices.json'}")

    sample = pd.DataFrame(all_data)[ "codes" ].head(3).tolist()
    print(f"Sample codes entries: {sample}")


if __name__ == "__main__":
    main()
