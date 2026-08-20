import argparse

import torch
from torch.utils.data import DataLoader

from src.datasets.nights import NIGHTSDataset
from src.models.transforms import get_dino_transform
from src.models.vit_baseline import DINOViTBaseline
from src.models.perceptual_similarity import PerceptualSimilarityModel


def parse_args():

    parser = argparse.ArgumentParser(
        description="Evaluate DINO ViT baseline on NIGHTS."
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="./data/nights",
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
        help=(
            "Fraction of the selected split to evaluate. "
            "For final test evaluation use 1.0."
        ),
    )

    return parser.parse_args()


def main():

    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("DINO ViT-B/16 NIGHTS baseline")
    print("=" * 60)

    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(device),
        )

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    transform = get_dino_transform()

    dataset = NIGHTSDataset(
        root_dir=args.data_root,
        split="test",
        transform=transform,
        subset_fraction=args.subset_fraction,
        seed=42,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Test triplets: {len(dataset)}")
    print(f"Batch size   : {args.batch_size}")

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    encoder = DINOViTBaseline(
        pretrained=True,
        freeze=False,
    )

    model = PerceptualSimilarityModel(
        encoder
    ).to(device)

    model.eval()

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    total = 0
    correct = 0

    distance_left_sum = 0.0
    distance_right_sum = 0.0

    with torch.inference_mode():

        for batch_idx, batch in enumerate(loader):

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

            batch_correct = (
                prediction == target
            ).sum().item()

            correct += batch_correct
            total += target.numel()

            distance_left_sum += (
                output["distance_left"]
                .sum()
                .item()
            )

            distance_right_sum += (
                output["distance_right"]
                .sum()
                .item()
            )

            if (
                batch_idx % 10 == 0
                or batch_idx == len(loader) - 1
            ):

                running_accuracy = (
                    correct / total
                )

                print(
                    f"Batch "
                    f"{batch_idx + 1:4d}/"
                    f"{len(loader):4d} "
                    f"| "
                    f"2AFC: "
                    f"{running_accuracy * 100:.2f}%"
                )

    # ---------------------------------------------------------
    # Final results
    # ---------------------------------------------------------

    accuracy = correct / total

    mean_left_distance = (
        distance_left_sum / total
    )

    mean_right_distance = (
        distance_right_sum / total
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"Correct   : {correct}")
    print(f"Total     : {total}")

    print(
        f"2AFC      : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Mean d(L) : "
        f"{mean_left_distance:.6f}"
    )

    print(
        f"Mean d(R) : "
        f"{mean_right_distance:.6f}"
    )


if __name__ == "__main__":
    main()