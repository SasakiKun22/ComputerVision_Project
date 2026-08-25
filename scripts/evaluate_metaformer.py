import argparse

import torch
from torch.utils.data import DataLoader

from src.datasets.nights import NIGHTSDataset
from src.models.transforms import get_dino_transform
from src.models.vit_metaformer import DINOViTMetaFormer
from src.models.perceptual_similarity import PerceptualSimilarityModel


# =====================================================================
# ARGUMENTS
# =====================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate DINO ViT-B/16 with "
            "MetaFormer / PoolFormer-style token mixer on NIGHTS."
        )
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
        help="Path to the trained MetaFormer checkpoint.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
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
            "Fraction of NIGHTS test split to evaluate. "
            "Use 1.0 for final evaluation."
        ),
    )

    return parser.parse_args()


# =====================================================================
# PARAMETER COUNT
# =====================================================================

def count_parameters(model):

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


# =====================================================================
# MAIN
# =====================================================================

def main():

    args = parse_args()

    # -----------------------------------------------------------------
    # Device
    # -----------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 72)
    print("NIGHTS METAFORMER EVALUATION")
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

    # =================================================================
    # LOAD CHECKPOINT METADATA FIRST
    # =================================================================

    print()
    print(
        f"Reading checkpoint: "
        f"{args.checkpoint}"
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
    )

    # -----------------------------------------------------------------
    # Recover architecture configuration
    # -----------------------------------------------------------------

    if "mode" not in checkpoint:
        raise ValueError(
            "Checkpoint does not contain 'mode'. "
            "Make sure it was created with the updated "
            "train_metaformer.py."
        )

    if "pool_size" not in checkpoint:
        raise ValueError(
            "Checkpoint does not contain 'pool_size'."
        )

    if "metaformer_layers" not in checkpoint:
        raise ValueError(
            "Checkpoint does not contain 'metaformer_layers'."
        )

    mode = checkpoint[
        "mode"
    ]

    pool_size = checkpoint[
        "pool_size"
    ]

    metaformer_layers = checkpoint[
        "metaformer_layers"
    ]

    checkpoint_epoch = checkpoint.get(
        "epoch",
        None,
    )

    checkpoint_val_accuracy = (
        checkpoint.get(
            "val_accuracy",
            None,
        )
    )

    # -----------------------------------------------------------------
    # Print checkpoint information
    # -----------------------------------------------------------------

    print()
    print(
        "Architecture information:"
    )

    print(
        "Mode:",
        mode,
    )

    print(
        "Pool size:",
        pool_size,
    )

    print(
        "MetaFormer layers:",
        metaformer_layers,
    )

    if checkpoint_epoch is not None:

        print(
            "Best epoch:",
            checkpoint_epoch,
        )

    if (
        checkpoint_val_accuracy
        is not None
    ):

        print(
            f"Checkpoint validation 2AFC: "
            f"{checkpoint_val_accuracy * 100:.2f}%"
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
    # BUILD MODEL
    # =================================================================

    print()
    print(
        "Building MetaFormer model..."
    )

    encoder = DINOViTMetaFormer(
        pretrained=True,
        freeze=False,
        pool_size=pool_size,
        metaformer_layers=(
            metaformer_layers
        ),
    )

    model = (
        PerceptualSimilarityModel(
            encoder
        )
        .to(device)
    )

    # -----------------------------------------------------------------
    # Show layer configuration
    # -----------------------------------------------------------------

    print()
    print(
        "Transformer token mixer configuration:"
    )

    for index, block in enumerate(
        encoder.get_transformer_blocks()
    ):

        print(
            f"Block {index:02d}: "
            f"{block.attn.__class__.__name__}"
        )

    # =================================================================
    # LOAD TRAINED WEIGHTS
    # =================================================================

    print()
    print(
        "Loading trained weights..."
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    print(
        "Checkpoint loaded successfully."
    )

    # =================================================================
    # PARAMETERS
    # =================================================================

    total_params, trainable_params = (
        count_parameters(model)
    )

    print()
    print(
        f"Total parameters    : "
        f"{total_params:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_params:,}"
    )

    # =================================================================
    # EVALUATION
    # =================================================================

    total = 0
    correct = 0

    distance_left_sum = 0.0
    distance_right_sum = 0.0

    print()
    print(
        "Starting evaluation..."
    )

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

            # ---------------------------------------------------------
            # Forward
            # ---------------------------------------------------------

            output = model(
                reference,
                left,
                right,
            )

            prediction = output[
                "prediction"
            ]

            # ---------------------------------------------------------
            # Accuracy
            # ---------------------------------------------------------

            batch_correct = (
                prediction
                == target
            ).sum().item()

            correct += (
                batch_correct
            )

            total += (
                target.numel()
            )

            # ---------------------------------------------------------
            # Mean distances
            # ---------------------------------------------------------

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

            # ---------------------------------------------------------
            # Progress
            # ---------------------------------------------------------

            if (
                (batch_idx + 1) % 10 == 0
                or
                batch_idx
                == len(loader) - 1
            ):

                running_accuracy = (
                    correct
                    / total
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
    # FINAL RESULTS
    # =================================================================

    accuracy = (
        correct
        / total
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
    print("=" * 72)
    print("RESULTS")
    print("=" * 72)

    print(
        f"Architecture : "
        f"MetaFormer {mode}"
    )

    print(
        f"Pool size    : "
        f"{pool_size}"
    )

    print(
        f"Modified     : "
        f"{len(metaformer_layers)}/12 layers"
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

    print(
        f"Parameters   : "
        f"{total_params:,}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()