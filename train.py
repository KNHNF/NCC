"""Train a small U-Net for matrix/fibre/void segmentation.

    python train.py --epochs 20 --batch-size 16 --subset 0   # full run
    python train.py --epochs 1 --subset 200                  # smoke test

Saves the best-Dice checkpoint to best_model.pth. Does not touch evaluation.py
or score_submission.py, those are the locked scorers, this script is the only
thing that should change while iterating.
"""

import argparse

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from dataset import VoidSegDataset, collect_samples, train_val_split

VOID_CLASS = 2
NUM_CLASSES = 3


def dice_void_score(pred_logits, mask):
    pred = pred_logits.argmax(dim=1)
    p, g = (pred == VOID_CLASS), (mask == VOID_CLASS)
    inter = (p & g).sum().item()
    denom = p.sum().item() + g.sum().item()
    return 1.0 if denom == 0 else 2.0 * inter / denom


def focal_dice_loss(logits, target, class_weights, gamma=2.0):
    ce = torch.nn.functional.cross_entropy(logits, target, weight=class_weights, reduction="none")
    pt = torch.exp(-ce)
    focal = ((1 - pt) ** gamma * ce).mean()

    probs = torch.softmax(logits, dim=1)[:, VOID_CLASS]
    void_target = (target == VOID_CLASS).float()
    inter = (probs * void_target).sum()
    dice = 1 - (2 * inter + 1) / (probs.sum() + void_target.sum() + 1)

    return focal + dice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--subset", type=int, default=0, help="0 = full dataset, N = first N samples (smoke test)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"Device: {args.device}")

    samples = collect_samples()
    if args.subset:
        samples = samples[: args.subset]
    train_samples, val_samples = train_val_split(samples)
    print(f"Train: {len(train_samples)}  Val: {len(val_samples)}")

    train_loader = DataLoader(VoidSegDataset(train_samples), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(VoidSegDataset(val_samples), batch_size=args.batch_size, shuffle=False)

    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=1,
        classes=NUM_CLASSES,
    ).to(args.device)

    # Matrix and fibre dominate pixel counts, void is rare and matters most,
    # weight it up so the loss does not just learn to predict matrix/fibre.
    class_weights = torch.tensor([1.0, 1.0, 5.0], device=args.device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_dice = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for img, mask, _, _ in train_loader:
            img, mask = img.to(args.device), mask.to(args.device)
            opt.zero_grad()
            logits = model(img)
            loss = focal_dice_loss(logits, mask, class_weights)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        model.eval()
        dices = []
        with torch.no_grad():
            for img, mask, _, _ in val_loader:
                img, mask = img.to(args.device), mask.to(args.device)
                logits = model(img)
                dices.append(dice_void_score(logits, mask))

        mean_dice = float(np.mean(dices)) if dices else 0.0
        print(f"Epoch {epoch + 1}/{args.epochs}  train_loss={total_loss / len(train_loader):.4f}  "
              f"val_dice_void={mean_dice:.4f}")

        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  saved best_model.pth (Dice {best_dice:.4f})")

    print(f"\nBest val Dice_void: {best_dice:.4f}  (gate is 0.8)")


if __name__ == "__main__":
    main()
