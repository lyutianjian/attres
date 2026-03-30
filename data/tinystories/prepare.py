"""
Prepare the TinyStories dataset for GPT-style language modeling.

This follows the same output format as the other nanoGPT data scripts:
- train.bin / val.bin: contiguous uint16 token streams
- meta.pkl: tokenizer metadata (at minimum vocab_size)

Examples:
    python data/tinystories/prepare.py
    python data/tinystories/prepare.py --max_train_examples=200000 --max_val_examples=2000
"""

import argparse
import os
import pickle

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


_ENC = None
_ENC_TOKENIZER = None


def process(example, tokenizer):
    global _ENC, _ENC_TOKENIZER
    if _ENC is None or _ENC_TOKENIZER != tokenizer:
        _ENC = tiktoken.get_encoding(tokenizer)
        _ENC_TOKENIZER = tokenizer

    ids = _ENC.encode_ordinary(example["text"])
    ids.append(_ENC.eot_token)
    return {"ids": ids, "len": len(ids)}


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare the TinyStories dataset")
    parser.add_argument("--dataset_name", type=str, default="roneneldan/TinyStories")
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="validation")
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--num_proc", type=int, default=8)
    parser.add_argument("--num_proc_load_dataset", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1024)
    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--max_val_examples", type=int, default=None)
    return parser.parse_args()


def write_split(dset, filename, num_shards):
    arr_len = np.sum(dset["len"], dtype=np.uint64)
    arr = np.memmap(filename, dtype=np.uint16, mode="w+", shape=(arr_len,))

    idx = 0
    for shard_idx in tqdm(range(num_shards), desc=f"writing {filename}"):
        shard = dset.shard(num_shards=num_shards, index=shard_idx, contiguous=True).with_format("numpy")
        if len(shard) == 0:
            continue
        arr_batch = np.concatenate(shard["ids"])
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)

    arr.flush()


if __name__ == "__main__":
    args = parse_args()
    if args.num_proc_load_dataset is None:
        args.num_proc_load_dataset = args.num_proc

    enc = tiktoken.get_encoding(args.tokenizer)
    data_dir = os.path.dirname(__file__)

    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        num_proc=args.num_proc_load_dataset,
    )

    split_dataset = {
        "train": dataset[args.train_split],
        "val": dataset[args.val_split],
    }

    if args.max_train_examples is not None:
        split_dataset["train"] = split_dataset["train"].select(range(min(args.max_train_examples, len(split_dataset["train"]))))
    if args.max_val_examples is not None:
        split_dataset["val"] = split_dataset["val"].select(range(min(args.max_val_examples, len(split_dataset["val"]))))

    tokenized = {}
    for split, dset in split_dataset.items():
        tokenized[split] = dset.map(
            process,
            fn_kwargs={"tokenizer": args.tokenizer},
            remove_columns=dset.column_names,
            desc=f"tokenizing {split}",
            num_proc=args.num_proc,
        )

    for split, dset in tokenized.items():
        filename = os.path.join(data_dir, f"{split}.bin")
        write_split(dset, filename, args.num_shards)
        total_tokens = np.sum(dset["len"], dtype=np.uint64)
        print(f"{split} has {total_tokens:,} tokens")

    meta = {
        "vocab_size": enc.n_vocab,
        "tokenizer": args.tokenizer,
        "dataset_name": args.dataset_name,
        "train_rows": len(split_dataset["train"]),
        "val_rows": len(split_dataset["val"]),
    }
    with open(os.path.join(data_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print("done")
