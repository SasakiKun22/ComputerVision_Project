import argparse
import csv
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from src.datasets.nights import NIGHTSDataset
from src.models.transforms import get_dino_transform
from src.models.vit_baseline import DINOViTBaseline
from src.models.perceptual_similarity import PerceptualSimilarityModel


# =====================================================================
# ARGUMENTS
# =====================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Fine-tune standard DINO ViT-B/16 on NIGHTS."
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="./data/nights",
        help="Path to NIGHTS dataset.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of triplets per batch.",
    )

    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="AdamW weight decay.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature used to convert distances into preference logits.",
    )

    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.5,
        help="Fraction of the NIGHTS training split to use.",
    )

    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.5,
        help="Fraction of the NIGHTS validation split to use.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/baseline_best.pth",
        help="Path for the best model checkpoint.",
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results/baseline",
        help="Directory where training curves and history are saved.",
    )

    return parser.parse_args()


# =====================================================================
# REPRODUCIBILITY
# =====================================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================
# RESULTS
# =====================================================================

def save_history_csv(history, results_dir):

    results_dir = Path(results_dir)

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = results_dir / "history.csv"

    with open(
        csv_path,
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "train_2afc",
            "val_2afc",
        ])

        for i in range(
            len(history["epoch"])
        ):

            writer.writerow([
                history["epoch"][i],
                history["train_loss"][i],
                history["val_loss"][i],
                history["train_2afc"][i],
                history["val_2afc"][i],
            ])


def plot_history(history, results_dir):

    results_dir = Path(results_dir)

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    epochs = history["epoch"]

    # ================================================================
    # LOSS CURVE
    # ================================================================

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        history["train_loss"],
        marker="o",
        label="Train",
    )

    plt.plot(
        epochs,
        history["val_loss"],
        marker="o",
        label="Validation",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")

    plt.title(
        "Training and validation loss"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir / "loss_curve.png",
        dpi=300,
    )

    plt.savefig(
        results_dir / "loss_curve.pdf",
    )

    plt.close()

    # ================================================================
    # 2AFC CURVE
    # ================================================================

    train_accuracy = [
        value * 100
        for value in history[
            "train_2afc"
        ]
    ]

    val_accuracy = [
        value * 100
        for value in history[
            "val_2afc"
        ]
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        train_accuracy,
        marker="o",
        label="Train",
    )

    plt.plot(
        epochs,
        val_accuracy,
        marker="o",
        label="Validation",
    )

    plt.xlabel("Epoch")
    plt.ylabel("2AFC accuracy (%)")

    plt.title(
        "Training and validation 2AFC accuracy"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir / "2afc_curve.png",
        dpi=300,
    )

    plt.savefig(
        results_dir / "2afc_curve.pdf",
    )

    plt.close()


# =====================================================================
# VALIDATION
# =====================================================================

@torch.inference_mode()
def evaluate(
    model,
    loader,
    device,
    criterion,
    temperature,
):

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:

        reference = batch[
            "reference"
        ].to(
            device,
            non_blocking=True,
        )

        left = batch[
            "left"
        ].to(
            device,
            non_blocking=True,
        )

        right = batch[
            "right"
        ].to(
            device,
            non_blocking=True,
        )

        target = batch[
            "target"
        ].long().to(
            device,
            non_blocking=True,
        )

        output = model(
            reference,
            left,
            right,
        )

        logits = (
            model.preference_logits(
                output["distance_left"],
                output["distance_right"],
                temperature=temperature,
            )
        )

        loss = criterion(
            logits,
            target,
        )

        prediction = output[
            "prediction"
        ]

        batch_size = (
            target.numel()
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        total_correct += (
            prediction
            == target
        ).sum().item()

        total_samples += (
            batch_size
        )

    average_loss = (
        total_loss
        / total_samples
    )

    accuracy = (
        total_correct
        / total_samples
    )

    return (
        average_loss,
        accuracy,
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    args = parse_args()

    # -----------------------------------------------------------------
    # Seed
    # -----------------------------------------------------------------

    set_seed(
        args.seed
    )

    # -----------------------------------------------------------------
    # Device
    # -----------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)

    print(
        "NIGHTS STANDARD ATTENTION BASELINE FINE-TUNING"
    )

    print("=" * 70)

    print(
        "Device:",
        device,
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    effective_batch_size = (
        args.batch_size
        * args.grad_accum
    )

    print(
        "Batch size:",
        args.batch_size,
    )

    print(
        "Gradient accumulation:",
        args.grad_accum,
    )

    print(
        "Effective batch size:",
        effective_batch_size,
    )

    print(
        "Learning rate:",
        args.lr,
    )

    print(
        "Epochs:",
        args.epochs,
    )

    # -----------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------

    transform = (
        get_dino_transform()
    )

    train_dataset = NIGHTSDataset(
        root_dir=args.data_root,
        split="train",
        transform=transform,
        subset_fraction=args.train_fraction,
        seed=args.seed,
    )

    val_dataset = NIGHTSDataset(
        root_dir=args.data_root,
        split="val",
        transform=transform,
        subset_fraction=args.val_fraction,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    print()

    print(
        "Train triplets:",
        len(train_dataset),
    )

    print(
        "Validation triplets:",
        len(val_dataset),
    )

    print(
        "Train batches:",
        len(train_loader),
    )

    print(
        "Validation batches:",
        len(val_loader),
    )

    # -----------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------

    print()

    print(
        "Loading pretrained DINO ViT-B/16..."
    )

    encoder = DINOViTBaseline(
        pretrained=True,
        freeze=False,
    )

    model = (
        PerceptualSimilarityModel(
            encoder
        )
        .to(device)
    )

    # -----------------------------------------------------------------
    # Loss
    # -----------------------------------------------------------------

    criterion = (
        nn.CrossEntropyLoss()
    )

    # -----------------------------------------------------------------
    # Optimizer
    # -----------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # -----------------------------------------------------------------
    # AMP
    # -----------------------------------------------------------------

    use_amp = (
        device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    # -----------------------------------------------------------------
    # Checkpoint directory
    # -----------------------------------------------------------------

    checkpoint_path = Path(
        args.checkpoint
    )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # Results directory
    # -----------------------------------------------------------------

    results_dir = Path(
        args.results_dir
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # Training history
    # -----------------------------------------------------------------

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_2afc": [],
        "val_2afc": [],
    }

    best_val_accuracy = 0.0
    best_epoch = 0

    # =================================================================
    # TRAINING LOOP
    # =================================================================

    for epoch in range(
        args.epochs
    ):

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        running_loss = 0.0
        running_correct = 0
        running_samples = 0

        print()
        print("=" * 70)

        print(
            f"Epoch "
            f"{epoch + 1}/"
            f"{args.epochs}"
        )

        print("=" * 70)

        # -------------------------------------------------------------
        # Train one epoch
        # -------------------------------------------------------------

        for batch_idx, batch in enumerate(
            train_loader
        ):

            reference = batch[
                "reference"
            ].to(
                device,
                non_blocking=True,
            )

            left = batch[
                "left"
            ].to(
                device,
                non_blocking=True,
            )

            right = batch[
                "right"
            ].to(
                device,
                non_blocking=True,
            )

            target = batch[
                "target"
            ].long().to(
                device,
                non_blocking=True,
            )

            # =========================================================
            # FORWARD
            # =========================================================

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp,
            ):

                output = model(
                    reference,
                    left,
                    right,
                )

                logits = (
                    model.preference_logits(
                        output[
                            "distance_left"
                        ],
                        output[
                            "distance_right"
                        ],
                        temperature=(
                            args.temperature
                        ),
                    )
                )

                loss = criterion(
                    logits,
                    target,
                )

                loss_for_backward = (
                    loss
                    / args.grad_accum
                )

            # =========================================================
            # BACKWARD
            # =========================================================

            scaler.scale(
                loss_for_backward
            ).backward()

            should_step = (
                (batch_idx + 1)
                % args.grad_accum
                == 0
            )

            last_batch = (
                batch_idx
                == len(train_loader) - 1
            )

            if (
                should_step
                or last_batch
            ):

                scaler.step(
                    optimizer
                )

                scaler.update()

                optimizer.zero_grad(
                    set_to_none=True
                )

            # =========================================================
            # STATISTICS
            # =========================================================

            batch_size = (
                target.numel()
            )

            running_loss += (
                loss.item()
                * batch_size
            )

            running_correct += (
                output["prediction"]
                == target
            ).sum().item()

            running_samples += (
                batch_size
            )

            # ---------------------------------------------------------
            # Console progress
            # ---------------------------------------------------------

            if (
                (batch_idx + 1)
                % 100
                == 0
                or last_batch
            ):

                current_loss = (
                    running_loss
                    / running_samples
                )

                current_accuracy = (
                    running_correct
                    / running_samples
                )

                print(
                    f"Batch "
                    f"{batch_idx + 1:4d}/"
                    f"{len(train_loader):4d}"
                    f" | "
                    f"Loss: "
                    f"{current_loss:.4f}"
                    f" | "
                    f"2AFC: "
                    f"{current_accuracy * 100:.2f}%"
                )

        # =============================================================
        # TRAIN RESULTS
        # =============================================================

        train_loss = (
            running_loss
            / running_samples
        )

        train_accuracy = (
            running_correct
            / running_samples
        )

        # =============================================================
        # VALIDATION
        # =============================================================

        val_loss, val_accuracy = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            criterion=criterion,
            temperature=args.temperature,
        )

        # =============================================================
        # PRINT EPOCH RESULTS
        # =============================================================

        print()

        print(
            f"Train loss : "
            f"{train_loss:.4f}"
        )

        print(
            f"Train 2AFC: "
            f"{train_accuracy * 100:.2f}%"
        )

        print(
            f"Val loss   : "
            f"{val_loss:.4f}"
        )

        print(
            f"Val 2AFC  : "
            f"{val_accuracy * 100:.2f}%"
        )

        # =============================================================
        # SAVE HISTORY
        # =============================================================

        history[
            "epoch"
        ].append(
            epoch + 1
        )

        history[
            "train_loss"
        ].append(
            train_loss
        )

        history[
            "val_loss"
        ].append(
            val_loss
        )

        history[
            "train_2afc"
        ].append(
            train_accuracy
        )

        history[
            "val_2afc"
        ].append(
            val_accuracy
        )

        # Save after every epoch.
        save_history_csv(
            history,
            results_dir,
        )

        plot_history(
            history,
            results_dir,
        )

        print(
            f"Training history saved to: "
            f"{results_dir}"
        )

        # =============================================================
        # SAVE BEST CHECKPOINT
        # =============================================================

        if (
            val_accuracy
            > best_val_accuracy
        ):

            best_val_accuracy = (
                val_accuracy
            )

            best_epoch = (
                epoch + 1
            )

            torch.save(
                {
                    "epoch":
                        epoch + 1,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "val_accuracy":
                        val_accuracy,

                    "val_loss":
                        val_loss,

                    "train_accuracy":
                        train_accuracy,

                    "train_loss":
                        train_loss,

                    "args":
                        vars(args),
                },
                checkpoint_path,
            )

            print(
                f"New best checkpoint saved: "
                f"{checkpoint_path}"
            )

    # =================================================================
    # FINAL SUMMARY
    # =================================================================

    print()
    print("=" * 70)

    print(
        "TRAINING COMPLETED"
    )

    print("=" * 70)

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best validation 2AFC: "
        f"{best_val_accuracy * 100:.2f}%"
    )

    print(
        f"Best checkpoint: "
        f"{checkpoint_path}"
    )

    print(
        f"Training results: "
        f"{results_dir}"
    )

    print()

    print(
        "Generated files:"
    )

    print(
        f"  {results_dir / 'history.csv'}"
    )

    print(
        f"  {results_dir / 'loss_curve.png'}"
    )

    print(
        f"  {results_dir / 'loss_curve.pdf'}"
    )

    print(
        f"  {results_dir / '2afc_curve.png'}"
    )

    print(
        f"  {results_dir / '2afc_curve.pdf'}"
    )


if __name__ == "__main__":
    main()