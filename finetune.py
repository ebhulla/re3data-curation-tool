"""Fine-tune all-MiniLM-L6-v2 so DUPLICATE name pairs score high and HIERARCHY / UNRELATED pairs score low.

Written by Claude under the one-time exception of 2026-10-08 -- FOR EKAM TO REVIEW after Fall Break.

Idea: the off-the-shelf model puts "X University" and "X University, Library" close together because they
share words. Training pairs with label 1 (duplicate) are pulled together and pairs with label 0 (hierarchy,
unrelated) are pushed apart -- hierarchy pairs act as the hard negatives.

Loss: OnlineContrastiveLoss. Within each batch it only uses the hardest examples (positives that are still
too far apart and negatives that are still too close), then applies a contrastive penalty with a margin.

Usage: python finetune.py --lr 2e-5 --epochs 3 --seed 1 --out models/ft_lr2e-5_s1
Then evaluate with: python baselines.py models/ft_lr2e-5_s1
"""
import argparse
import random

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="sentence-transformers/all-MiniLM-L6-v2")
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--out", required=True)
args = ap.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

train = pd.read_csv("data/pairs_train.csv")
print(f"training pairs: {len(train)}  (duplicates: {train.is_duplicate.sum()}, others: {(1 - train.is_duplicate).sum()})")
dataset = Dataset.from_dict({
    "sentence1": train.name_a.tolist(),
    "sentence2": train.name_b.tolist(),
    "label": train.is_duplicate.astype(float).tolist(),
})

model = SentenceTransformer(args.base)
loss = losses.OnlineContrastiveLoss(model)
training_args = SentenceTransformerTrainingArguments(
    output_dir=args.out + "_checkpoints",
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch,
    learning_rate=args.lr,
    warmup_ratio=0.1,
    seed=args.seed,
    save_strategy="no",
    report_to="none",
    logging_steps=10,
)
SentenceTransformerTrainer(model=model, args=training_args, train_dataset=dataset, loss=loss).train()
model.save(args.out)
print("saved to", args.out)
