import argparse

import torch
from torch.utils.data import DataLoader

from src.datasets.nights import NIGHTSDataset
from src.models.transforms import get_dino_transform
from src.models.vit_efficient import DINOViTEfficient
from src.models.perceptual_similarity import PerceptualSimilarityModel


# =====================================================================
# ARGUMENTS
# =====================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Evaluate SRA DINO ViT on NIGHTS."
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="./data/nights",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "all",
            "last_half",
        ],
        required=True,
    )

    parser.add_argument(
        "--sr-ratio",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--subset-fraction",
        type=float,
        default=1.0,
    )

    return parser.parse_args()


# =====================================================================
# EFFICIENT LAYERS
# =====================================================================

def get_efficient_layers(mode):

    if mode == "all":
        return list(range(12))

    if mode == "last_half":
        return list(range(6, 12))

    raise ValueError(
        f"Unsupported mode: {mode}"
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    args = parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    efficient_layers = (
        get_efficient_layers(
            args.mode
        )
    )

    print("=" * 70)
    print("NIGHTS SRA EVALUATION")
    print("=" * 70)

    print("Device:", device)

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print(
        "Mode:",
        args.mode,
    )

    print(
        "SR ratio:",
        args.sr_ratio,
    )

    print(
        "Efficient layers:",
        efficient_layers,
    )

    print(
        "Checkpoint:",
        args.checkpoint,
    )

    # =================================================================
    # DATASET
    # =================================================================

    transform = (
        get_dino_transform()
    )

    dataset = NIGHTSDataset(
        root_dir=args.data_root,
        split="test",
        transform=transform,
        subset_fraction=(
            args.subset_fraction
        ),
        seed=42,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    print()
    print(
        "Test triplets:",
        len(dataset),
    )

    print(
        "Batch size:",
        args.batch_size,
    )

    # =================================================================
    # MODEL
    # =================================================================

    print()
    print(
        "Building SRA model..."
    )

    encoder = DINOViTEfficient(
        pretrained=True,
        freeze=False,
        sr_ratio=args.sr_ratio,
        efficient_layers=(
            efficient_layers
        ),
    )

    model = (
        PerceptualSimilarityModel(
            encoder
        )
        .to(device)
    )

    # =================================================================
    # CHECKPOINT
    # =================================================================

    print(
        f"Loading checkpoint: "
        f"{args.checkpoint}"
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    print(
        "Checkpoint epoch:",
        checkpoint["epoch"],
    )

    print(
        f"Checkpoint validation 2AFC: "
        f"{checkpoint['val_accuracy'] * 100:.2f}%"
    )

    # =================================================================
    # EVALUATION
    # =================================================================

    total = 0
    correct = 0

    distance_left_sum = 0.0
    distance_right_sum = 0.0

    with torch.inference_mode():

        for batch_idx, batch in enumerate(
            loader
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

            output = model(
                reference,
                left,
                right,
            )

            prediction = output[
                "prediction"
            ]

            correct += (
                prediction
                == target
            ).sum().item()

            total += (
                target.numel()
            )

            distance_left_sum += (
                output[
                    "distance_left"
                ]
                .sum()
                .item()
            )

            distance_right_sum += (
                output[
                    "distance_right"
                ]
                .sum()
                .item()
            )

            if (
                (batch_idx + 1) % 10 == 0
                or batch_idx
                == len(loader) - 1
            ):

                running_accuracy = (
                    correct / total
                )

                print(
                    f"Batch "
                    f"{batch_idx + 1:4d}/"
                    f"{len(loader):4d}"
                    f" | "
                    f"2AFC: "
                    f"{running_accuracy * 100:.2f}%"
                )

    # =================================================================
    # RESULTS
    # =================================================================

    accuracy = (
        correct / total
    )

    mean_left_distance = (
        distance_left_sum
        / total
    )

    mean_right_distance = (
        distance_right_sum
        / total
    )

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"Architecture : "
        f"SRA {args.mode}"
    )

    print(
        f"SR ratio     : "
        f"{args.sr_ratio}"
    )

    print(
        f"Correct      : "
        f"{correct}"
    )

    print(
        f"Total        : "
        f"{total}"
    )

    print(
        f"Test 2AFC    : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Mean d(L)    : "
        f"{mean_left_distance:.6f}"
    )

    print(
        f"Mean d(R)    : "
        f"{mean_right_distance:.6f}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()