import argparse

import torch
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
            "Evaluate DINO ViT-B/16 with "
            "Mixture-of-Heads Attention on NIGHTS."
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
    )

    return parser.parse_args()


# =====================================================================
# PARAMETERS
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

    return (
        total,
        trainable,
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

    print("=" * 72)
    print(
        "NIGHTS MIXTURE-OF-HEADS EVALUATION"
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

    # =================================================================
    # CHECKPOINT
    # =================================================================

    print()
    print(
        "Reading checkpoint:",
        args.checkpoint,
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
    )

    # -----------------------------------------------------------------
    # Architecture metadata
    # -----------------------------------------------------------------

    required_metadata = [
        "mode",
        "shared_heads",
        "routed_heads",
        "moh_layers",
    ]

    for key in required_metadata:

        if key not in checkpoint:

            raise ValueError(
                f"Checkpoint does not contain "
                f"required metadata '{key}'."
            )

    mode = checkpoint[
        "mode"
    ]

    shared_heads = checkpoint[
        "shared_heads"
    ]

    routed_heads = checkpoint[
        "routed_heads"
    ]

    moh_layers = checkpoint[
        "moh_layers"
    ]

    aux_lambda = checkpoint.get(
        "aux_lambda",
        None,
    )

    print()
    print(
        "Architecture information:"
    )

    print(
        "Mode:",
        mode,
    )

    print(
        "MoH layers:",
        moh_layers,
    )

    print(
        "Shared heads:",
        shared_heads,
    )

    print(
        "Routed heads:",
        routed_heads,
    )

    print(
        "Active heads:",
        shared_heads
        + routed_heads,
        "/ 12",
    )

    if aux_lambda is not None:

        print(
            "Training aux lambda:",
            aux_lambda,
        )

    if "epoch" in checkpoint:

        print(
            "Best epoch:",
            checkpoint[
                "epoch"
            ],
        )

    if "val_accuracy" in checkpoint:

        print(
            f"Checkpoint validation 2AFC: "
            f"{checkpoint['val_accuracy'] * 100:.2f}%"
        )

    # =================================================================
    # DATASET
    # =================================================================

    dataset = NIGHTSDataset(
        root_dir=args.data_root,
        split="test",
        transform=(
            get_dino_transform()
        ),
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
        "Building MoH model..."
    )

    encoder = DINOViTMoH(
        pretrained=True,
        freeze=False,
        shared_heads=(
            shared_heads
        ),
        routed_heads=(
            routed_heads
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
    # Layer configuration
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

    # =================================================================
    # LOAD WEIGHTS
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

    (
        total_params,
        trainable_params,
    ) = count_parameters(
        model
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

    correct = 0
    total = 0

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

            correct += (
                prediction
                == target
            ).sum().item()

            total += (
                target.numel()
            )

            # ---------------------------------------------------------
            # Distances
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
                (batch_idx + 1)
                % 10
                == 0
                or batch_idx
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
    # RESULTS
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
    print(
        "RESULTS"
    )
    print("=" * 72)

    print(
        f"Architecture : "
        f"MoH {mode}"
    )

    print(
        f"Modified     : "
        f"{len(moh_layers)}/12 layers"
    )

    print(
        f"Shared heads : "
        f"{shared_heads}"
    )

    print(
        f"Routed heads : "
        f"{routed_heads}"
    )

    print(
        f"Active heads : "
        f"{shared_heads + routed_heads}/12"
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