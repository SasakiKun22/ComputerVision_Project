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
from src.models.vit_moh import DINOViTMoH
from src.models.perceptual_similarity import PerceptualSimilarityModel


# =====================================================================
# ARGUMENTS
# =====================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune DINO ViT-B/16 with "
            "Mixture-of-Heads Attention on NIGHTS."
        )
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="./data/nights",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    # -----------------------------------------------------------------
    # MoH configuration
    # -----------------------------------------------------------------

    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "all",
            "last_half",
        ],
        default="all",
        help=(
            "'all' replaces all 12 attention blocks with MoH; "
            "'last_half' replaces blocks 6-11 only."
        ),
    )

    parser.add_argument(
        "--shared-heads",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--routed-heads",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--aux-lambda",
        type=float,
        default=0.01,
        help=(
            "Weight of the MoH load-balancing auxiliary loss."
        ),
    )

    # -----------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
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
# SELECT MoH LAYERS
# =====================================================================

def get_moh_layers(mode):

    if mode == "all":

        return list(range(12))

    if mode == "last_half":

        return list(range(6, 12))

    raise ValueError(
        f"Unsupported mode: {mode}"
    )


# =====================================================================
# OUTPUT PATHS
# =====================================================================

def get_output_paths(args):

    experiment_name = (
        f"moh_{args.mode}"
        f"_shared{args.shared_heads}"
        f"_routed{args.routed_heads}"
    )

    if args.checkpoint is None:

        checkpoint_path = (
            Path("./checkpoints")
            / f"{experiment_name}_best.pth"
        )

    else:

        checkpoint_path = Path(
            args.checkpoint
        )

    if args.results_dir is None:

        results_dir = (
            Path("./results")
            / experiment_name
        )

    else:

        results_dir = Path(
            args.results_dir
        )

    return (
        checkpoint_path,
        results_dir,
    )


# =====================================================================
# SAVE HISTORY
# =====================================================================

def save_history_csv(
    history,
    results_dir,
):

    results_dir = Path(
        results_dir
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        results_dir
        / "history.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "train_loss",
            "train_aux_loss",
            "train_total_loss",
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
                history["train_aux_loss"][i],
                history["train_total_loss"][i],
                history["val_loss"][i],
                history["train_2afc"][i],
                history["val_2afc"][i],
            ])


# =====================================================================
# PLOTS
# =====================================================================

def plot_history(
    history,
    results_dir,
):

    results_dir = Path(
        results_dir
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    epochs = history["epoch"]

    # -----------------------------------------------------------------
    # Cross-entropy loss
    #
    # Keep this comparable to baseline / SRA / MetaFormer.
    # -----------------------------------------------------------------

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
    plt.ylabel(
        "Cross-entropy loss"
    )

    plt.title(
        "MoH training and validation loss"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir
        / "loss_curve.png",
        dpi=300,
    )

    plt.savefig(
        results_dir
        / "loss_curve.pdf",
    )

    plt.close()

    # -----------------------------------------------------------------
    # 2AFC accuracy
    # -----------------------------------------------------------------

    train_accuracy = [
        value * 100
        for value
        in history["train_2afc"]
    ]

    val_accuracy = [
        value * 100
        for value
        in history["val_2afc"]
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
    plt.ylabel(
        "2AFC accuracy (%)"
    )

    plt.title(
        "MoH training and validation 2AFC accuracy"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir
        / "2afc_curve.png",
        dpi=300,
    )

    plt.savefig(
        results_dir
        / "2afc_curve.pdf",
    )

    plt.close()

    # -----------------------------------------------------------------
    # Auxiliary loss
    # -----------------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        history[
            "train_aux_loss"
        ],
        marker="o",
        label="MoH auxiliary loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel(
        "Load-balancing loss"
    )

    plt.title(
        "MoH routing auxiliary loss"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir
        / "aux_loss_curve.png",
        dpi=300,
    )

    plt.savefig(
        results_dir
        / "aux_loss_curve.pdf",
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
                output[
                    "distance_left"
                ],
                output[
                    "distance_right"
                ],
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

    set_seed(
        args.seed
    )

    # =================================================================
    # DEVICE
    # =================================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # =================================================================
    # CONFIGURATION
    # =================================================================

    moh_layers = (
        get_moh_layers(
            args.mode
        )
    )

    (
        checkpoint_path,
        results_dir,
    ) = get_output_paths(
        args
    )

    print("=" * 72)
    print(
        "NIGHTS MIXTURE-OF-HEADS FINE-TUNING"
    )
    print("=" * 72)

    print(
        "Device:",
        device,
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print()

    print(
        "Mode:",
        args.mode,
    )

    print(
        "MoH layers:",
        moh_layers,
    )

    print(
        "Shared heads:",
        args.shared_heads,
    )

    print(
        "Routed heads:",
        args.routed_heads,
    )

    print(
        "Active heads:",
        args.shared_heads
        + args.routed_heads,
        "/ 12",
    )

    print(
        "Aux lambda:",
        args.aux_lambda,
    )

    print()

    print(
        "Epochs:",
        args.epochs,
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
        args.batch_size
        * args.grad_accum,
    )

    print(
        "Learning rate:",
        args.lr,
    )

    print(
        "Train fraction:",
        args.train_fraction,
    )

    print(
        "Validation fraction:",
        args.val_fraction,
    )

    print(
        "Checkpoint:",
        checkpoint_path,
    )

    print(
        "Results:",
        results_dir,
    )

    # =================================================================
    # DATA
    # =================================================================

    transform = (
        get_dino_transform()
    )

    train_dataset = NIGHTSDataset(
        root_dir=args.data_root,
        split="train",
        transform=transform,
        subset_fraction=(
            args.train_fraction
        ),
        seed=args.seed,
    )

    val_dataset = NIGHTSDataset(
        root_dir=args.data_root,
        split="val",
        transform=transform,
        subset_fraction=(
            args.val_fraction
        ),
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

    # =================================================================
    # MODEL
    # =================================================================

    print()
    print(
        "Loading pretrained DINO ViT-B/16..."
    )

    encoder = DINOViTMoH(
        pretrained=True,
        freeze=False,
        shared_heads=(
            args.shared_heads
        ),
        routed_heads=(
            args.routed_heads
        ),
        moh_layers=(
            moh_layers
        ),
    )

    model = (
        PerceptualSimilarityModel(
            encoder
        )
        .to(device)
    )

    # -----------------------------------------------------------------
    # Architecture check
    # -----------------------------------------------------------------

    print()
    print(
        "Transformer attention configuration:"
    )

    for index, block in enumerate(
        encoder.get_transformer_blocks()
    ):

        print(
            f"Block {index:02d}: "
            f"{block.attn.__class__.__name__}"
        )

    # -----------------------------------------------------------------
    # Parameters
    # -----------------------------------------------------------------

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()

    print(
        f"Total parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    # =================================================================
    # LOSS / OPTIMIZER
    # =================================================================

    criterion = (
        nn.CrossEntropyLoss()
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=(
            args.weight_decay
        ),
    )

    # =================================================================
    # AMP
    # =================================================================

    use_amp = (
        device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    # =================================================================
    # OUTPUT
    # =================================================================

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =================================================================
    # HISTORY
    # =================================================================

    history = {
        "epoch": [],
        "train_loss": [],
        "train_aux_loss": [],
        "train_total_loss": [],
        "val_loss": [],
        "train_2afc": [],
        "val_2afc": [],
    }

    best_val_accuracy = 0.0
    best_epoch = 0

    # =================================================================
    # TRAINING
    # =================================================================

    for epoch in range(
        args.epochs
    ):

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        running_ce_loss = 0.0
        running_aux_loss = 0.0
        running_total_loss = 0.0

        running_correct = 0
        running_samples = 0

        print()
        print("=" * 72)

        print(
            f"Epoch "
            f"{epoch + 1}/"
            f"{args.epochs}"
        )

        print("=" * 72)

        # -------------------------------------------------------------
        # TRAIN
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
                device_type=device.type,
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

                ce_loss = criterion(
                    logits,
                    target,
                )

                # -----------------------------------------------------
                # MoH auxiliary routing loss
                # -----------------------------------------------------

                aux_loss = (
                    encoder
                    .get_moh_aux_loss()
                )

                if aux_loss is None:

                    aux_loss = torch.zeros(
                        (),
                        device=device,
                        dtype=ce_loss.dtype,
                    )

                total_loss = (
                    ce_loss
                    +
                    args.aux_lambda
                    * aux_loss
                )

                loss_for_backward = (
                    total_loss
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

            running_ce_loss += (
                ce_loss.item()
                * batch_size
            )

            running_aux_loss += (
                aux_loss.item()
                * batch_size
            )

            running_total_loss += (
                total_loss.item()
                * batch_size
            )

            running_correct += (
                output[
                    "prediction"
                ]
                == target
            ).sum().item()

            running_samples += (
                batch_size
            )

            # ---------------------------------------------------------
            # Progress
            # ---------------------------------------------------------

            if (
                (batch_idx + 1)
                % 100
                == 0
                or last_batch
            ):

                current_ce = (
                    running_ce_loss
                    / running_samples
                )

                current_aux = (
                    running_aux_loss
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
                    f"CE: "
                    f"{current_ce:.4f}"
                    f" | "
                    f"Aux: "
                    f"{current_aux:.4f}"
                    f" | "
                    f"2AFC: "
                    f"{current_accuracy * 100:.2f}%"
                )

        # =================================================================
        # TRAIN RESULTS
        # =================================================================

        train_loss = (
            running_ce_loss
            / running_samples
        )

        train_aux_loss = (
            running_aux_loss
            / running_samples
        )

        train_total_loss = (
            running_total_loss
            / running_samples
        )

        train_accuracy = (
            running_correct
            / running_samples
        )

        # =================================================================
        # VALIDATION
        # =================================================================

        (
            val_loss,
            val_accuracy,
        ) = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            criterion=criterion,
            temperature=(
                args.temperature
            ),
        )

        # =================================================================
        # PRINT
        # =================================================================

        print()

        print(
            f"Train CE loss    : "
            f"{train_loss:.4f}"
        )

        print(
            f"Train aux loss   : "
            f"{train_aux_loss:.4f}"
        )

        print(
            f"Train total loss : "
            f"{train_total_loss:.4f}"
        )

        print(
            f"Train 2AFC       : "
            f"{train_accuracy * 100:.2f}%"
        )

        print(
            f"Val CE loss      : "
            f"{val_loss:.4f}"
        )

        print(
            f"Val 2AFC         : "
            f"{val_accuracy * 100:.2f}%"
        )

        # =================================================================
        # HISTORY
        # =================================================================

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
            "train_aux_loss"
        ].append(
            train_aux_loss
        )

        history[
            "train_total_loss"
        ].append(
            train_total_loss
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

        save_history_csv(
            history,
            results_dir,
        )

        plot_history(
            history,
            results_dir,
        )

        # =================================================================
        # BEST CHECKPOINT
        # =================================================================

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

                    "train_accuracy":
                        train_accuracy,

                    "train_loss":
                        train_loss,

                    "train_aux_loss":
                        train_aux_loss,

                    "train_total_loss":
                        train_total_loss,

                    "val_accuracy":
                        val_accuracy,

                    "val_loss":
                        val_loss,

                    "mode":
                        args.mode,

                    "shared_heads":
                        args.shared_heads,

                    "routed_heads":
                        args.routed_heads,

                    "moh_layers":
                        moh_layers,

                    "aux_lambda":
                        args.aux_lambda,

                    "total_parameters":
                        total_parameters,

                    "trainable_parameters":
                        trainable_parameters,

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
    # SUMMARY
    # =================================================================

    print()
    print("=" * 72)
    print(
        "TRAINING COMPLETED"
    )
    print("=" * 72)

    print(
        f"Mode: "
        f"{args.mode}"
    )

    print(
        f"MoH layers: "
        f"{moh_layers}"
    )

    print(
        f"Shared heads: "
        f"{args.shared_heads}"
    )

    print(
        f"Routed heads: "
        f"{args.routed_heads}"
    )

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
        f"Results: "
        f"{results_dir}"
    )


if __name__ == "__main__":
    main()