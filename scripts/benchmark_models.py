import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.nights import NIGHTSDataset

from src.models.transforms import (
    get_dino_transform,
)

from src.models.vit_baseline import (
    DINOViTBaseline,
)

from src.models.vit_efficient import (
    DINOViTEfficient,
)

from src.models.vit_metaformer import (
    DINOViTMetaFormer,
)

from src.models.perceptual_similarity import (
    PerceptualSimilarityModel,
)


# =====================================================================
# ARGUMENTS
# =====================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "GPU inference benchmark for NIGHTS "
            "perceptual similarity models."
        )
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="./data/nights",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "baseline",
            "sra_all",
            "sra_half",
            "metaformer_all",
            "metaformer_half",
        ],
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
        "--warmup-batches",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--benchmark-batches",
        type=int,
        default=40,
    )

    # -----------------------------------------------------------------
    # Fallback values.
    #
    # Normally SRA / MetaFormer configuration is read directly
    # from the checkpoint.
    # -----------------------------------------------------------------

    parser.add_argument(
        "--sr-ratio",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--pool-size",
        type=int,
        default=3,
    )

    # -----------------------------------------------------------------
    # Precision
    # -----------------------------------------------------------------

    parser.add_argument(
        "--amp",
        action="store_true",
        help="Run benchmark using FP16 autocast.",
    )

    # -----------------------------------------------------------------
    # Optional accuracy value.
    #
    # This does NOT recompute 2AFC.
    # It simply stores the already measured test accuracy in the CSV,
    # which is useful for final plots/tables.
    # -----------------------------------------------------------------

    parser.add_argument(
        "--test-2afc",
        type=float,
        default=None,
        help=(
            "Previously measured test 2AFC in percentage points "
            "(e.g. 93.59)."
        ),
    )

    # -----------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "./results/benchmark/"
            "inference_benchmark.csv"
        ),
    )

    return parser.parse_args()


# =====================================================================
# PARAMETER COUNT
# =====================================================================

def count_parameters(model):

    total = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    return total, trainable


# =====================================================================
# LOAD CHECKPOINT + BUILD CORRECT ARCHITECTURE
# =====================================================================

def build_model(
    args,
    device,
):

    # -----------------------------------------------------------------
    # Load checkpoint metadata on CPU first.
    # -----------------------------------------------------------------

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
    )

    architecture_info = {}

    # =================================================================
    # STANDARD BASELINE
    # =================================================================

    if args.model == "baseline":

        encoder = DINOViTBaseline(
            pretrained=True,
            freeze=False,
        )

        architecture_info = {
            "architecture":
                "standard_attention",

            "efficient_layers":
                0,

            "reduction":
                "-",
        }

    # =================================================================
    # SRA FULL
    # =================================================================

    elif args.model == "sra_all":

        sr_ratio = checkpoint.get(
            "sr_ratio",
            args.sr_ratio,
        )

        efficient_layers = (
            checkpoint.get(
                "efficient_layers",
                list(range(12)),
            )
        )

        encoder = DINOViTEfficient(
            pretrained=True,
            freeze=False,
            sr_ratio=sr_ratio,
            efficient_layers=(
                efficient_layers
            ),
        )

        architecture_info = {
            "architecture":
                "SRA",

            "efficient_layers":
                len(efficient_layers),

            "reduction":
                f"sr_ratio={sr_ratio}",
        }

    # =================================================================
    # SRA HALF
    # =================================================================

    elif args.model == "sra_half":

        sr_ratio = checkpoint.get(
            "sr_ratio",
            args.sr_ratio,
        )

        efficient_layers = (
            checkpoint.get(
                "efficient_layers",
                list(range(6, 12)),
            )
        )

        encoder = DINOViTEfficient(
            pretrained=True,
            freeze=False,
            sr_ratio=sr_ratio,
            efficient_layers=(
                efficient_layers
            ),
        )

        architecture_info = {
            "architecture":
                "SRA",

            "efficient_layers":
                len(efficient_layers),

            "reduction":
                f"sr_ratio={sr_ratio}",
        }

    # =================================================================
    # METAFORMER FULL
    # =================================================================

    elif args.model == "metaformer_all":

        pool_size = checkpoint.get(
            "pool_size",
            args.pool_size,
        )

        metaformer_layers = (
            checkpoint.get(
                "metaformer_layers",
                list(range(12)),
            )
        )

        encoder = DINOViTMetaFormer(
            pretrained=True,
            freeze=False,
            pool_size=pool_size,
            metaformer_layers=(
                metaformer_layers
            ),
        )

        architecture_info = {
            "architecture":
                "MetaFormerPooling",

            "efficient_layers":
                len(metaformer_layers),

            "reduction":
                f"pool_size={pool_size}",
        }

    # =================================================================
    # METAFORMER HALF
    # =================================================================

    elif args.model == "metaformer_half":

        pool_size = checkpoint.get(
            "pool_size",
            args.pool_size,
        )

        metaformer_layers = (
            checkpoint.get(
                "metaformer_layers",
                list(range(6, 12)),
            )
        )

        encoder = DINOViTMetaFormer(
            pretrained=True,
            freeze=False,
            pool_size=pool_size,
            metaformer_layers=(
                metaformer_layers
            ),
        )

        architecture_info = {
            "architecture":
                "MetaFormerPooling",

            "efficient_layers":
                len(metaformer_layers),

            "reduction":
                f"pool_size={pool_size}",
        }

    else:

        raise ValueError(
            f"Unsupported model: "
            f"{args.model}"
        )

    # =================================================================
    # DREAMSIM-LIKE PIPELINE
    # =================================================================

    model = (
        PerceptualSimilarityModel(
            encoder
        )
        .to(device)
    )

    # =================================================================
    # LOAD TRAINED WEIGHTS
    # =================================================================

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    return (
        model,
        architecture_info,
        checkpoint,
    )


# =====================================================================
# CSV
# =====================================================================

def append_result(
    output_path,
    result,
):

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_exists = (
        output_path.exists()
    )

    fieldnames = [
        "model",
        "architecture",
        "efficient_layers",
        "reduction",
        "batch_size",
        "amp",
        "benchmark_batches",
        "num_triplets",
        "total_gpu_time_ms",
        "ms_per_batch",
        "ms_per_triplet",
        "triplets_per_second",
        "peak_vram_mb",
        "parameters",
        "trainable_parameters",
        "test_2afc",
    ]

    with open(
        output_path,
        "a",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        if not file_exists:

            writer.writeheader()

        writer.writerow(
            result
        )


# =====================================================================
# MOVE BATCH TO GPU
# =====================================================================

def prepare_batch(
    batch,
    device,
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

    return (
        reference,
        left,
        right,
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    args = parse_args()

    # =================================================================
    # CUDA CHECK
    # =================================================================

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is required for this benchmark."
        )

    device = torch.device(
        "cuda"
    )

    print("=" * 72)

    print(
        "GPU INFERENCE BENCHMARK"
    )

    print("=" * 72)

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    print(
        "Model:",
        args.model,
    )

    print(
        "Checkpoint:",
        args.checkpoint,
    )

    print(
        "Batch size:",
        args.batch_size,
    )

    print(
        "AMP:",
        args.amp,
    )

    print(
        "Warm-up batches:",
        args.warmup_batches,
    )

    print(
        "Requested benchmark batches:",
        args.benchmark_batches,
    )

    # =================================================================
    # DATASET
    # =================================================================

    dataset = NIGHTSDataset(
        root_dir=args.data_root,
        split="test",
        transform=get_dino_transform(),
        subset_fraction=1.0,
        seed=42,
    )

    # drop_last ensures every timed batch has exactly the same size.
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    print()

    print(
        "Test triplets:",
        len(dataset),
    )

    print(
        "Full batches available:",
        len(loader),
    )

    # =================================================================
    # MODEL
    # =================================================================

    print()
    print(
        "Loading model..."
    )

    (
        model,
        architecture_info,
        checkpoint,
    ) = build_model(
        args,
        device,
    )

    total_params, trainable_params = (
        count_parameters(
            model
        )
    )

    print()

    print(
        "Architecture:",
        architecture_info[
            "architecture"
        ],
    )

    print(
        "Efficient layers:",
        architecture_info[
            "efficient_layers"
        ],
    )

    print(
        "Configuration:",
        architecture_info[
            "reduction"
        ],
    )

    print(
        f"Parameters: "
        f"{total_params:,}"
    )

    if "epoch" in checkpoint:

        print(
            "Checkpoint epoch:",
            checkpoint["epoch"],
        )

    if "val_accuracy" in checkpoint:

        print(
            f"Checkpoint val 2AFC: "
            f"{checkpoint['val_accuracy'] * 100:.2f}%"
        )

    # =================================================================
    # WARM-UP
    # =================================================================

    print()
    print(
        "Starting GPU warm-up..."
    )

    loader_iterator = iter(
        loader
    )

    with torch.inference_mode():

        for warmup_index in range(
            args.warmup_batches
        ):

            try:

                batch = next(
                    loader_iterator
                )

            except StopIteration:

                raise RuntimeError(
                    "Not enough batches for "
                    "the requested warm-up."
                )

            (
                reference,
                left,
                right,
            ) = prepare_batch(
                batch,
                device,
            )

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=args.amp,
            ):

                model(
                    reference,
                    left,
                    right,
                )

    torch.cuda.synchronize()

    # -----------------------------------------------------------------
    # Free references from warm-up.
    # -----------------------------------------------------------------

    del reference
    del left
    del right
    del batch

    torch.cuda.synchronize()

    # =================================================================
    # RESET PEAK VRAM
    # =================================================================

    torch.cuda.reset_peak_memory_stats(
        device
    )

    # =================================================================
    # BENCHMARK
    # =================================================================

    print()
    print(
        "Starting benchmark..."
    )

    total_gpu_time_ms = 0.0
    measured_batches = 0
    measured_triplets = 0

    with torch.inference_mode():

        for benchmark_index in range(
            args.benchmark_batches
        ):

            try:

                batch = next(
                    loader_iterator
                )

            except StopIteration:

                break

            (
                reference,
                left,
                right,
            ) = prepare_batch(
                batch,
                device,
            )

            # ---------------------------------------------------------
            # Synchronize data transfer before timing.
            # ---------------------------------------------------------

            torch.cuda.synchronize()

            start_event = torch.cuda.Event(
                enable_timing=True
            )

            end_event = torch.cuda.Event(
                enable_timing=True
            )

            start_event.record()

            # ---------------------------------------------------------
            # Timed GPU inference
            # ---------------------------------------------------------

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=args.amp,
            ):

                model(
                    reference,
                    left,
                    right,
                )

            end_event.record()

            torch.cuda.synchronize()

            elapsed_ms = (
                start_event.elapsed_time(
                    end_event
                )
            )

            total_gpu_time_ms += (
                elapsed_ms
            )

            measured_batches += 1

            measured_triplets += (
                reference.shape[0]
            )

    if measured_batches == 0:

        raise RuntimeError(
            "No benchmark batches were measured."
        )

    # =================================================================
    # RESULTS
    # =================================================================

    ms_per_batch = (
        total_gpu_time_ms
        / measured_batches
    )

    ms_per_triplet = (
        total_gpu_time_ms
        / measured_triplets
    )

    total_gpu_time_seconds = (
        total_gpu_time_ms
        / 1000.0
    )

    throughput = (
        measured_triplets
        / total_gpu_time_seconds
    )

    peak_vram_mb = (
        torch.cuda.max_memory_allocated(
            device
        )
        / (1024 ** 2)
    )

    result = {

        "model":
            args.model,

        "architecture":
            architecture_info[
                "architecture"
            ],

        "efficient_layers":
            architecture_info[
                "efficient_layers"
            ],

        "reduction":
            architecture_info[
                "reduction"
            ],

        "batch_size":
            args.batch_size,

        "amp":
            args.amp,

        "benchmark_batches":
            measured_batches,

        "num_triplets":
            measured_triplets,

        "total_gpu_time_ms":
            total_gpu_time_ms,

        "ms_per_batch":
            ms_per_batch,

        "ms_per_triplet":
            ms_per_triplet,

        "triplets_per_second":
            throughput,

        "peak_vram_mb":
            peak_vram_mb,

        "parameters":
            total_params,

        "trainable_parameters":
            trainable_params,

        "test_2afc":
            args.test_2afc,
    }

    # =================================================================
    # PRINT RESULTS
    # =================================================================

    print()
    print("=" * 72)
    print("BENCHMARK RESULTS")
    print("=" * 72)

    print(
        f"Model                : "
        f"{args.model}"
    )

    print(
        f"Measured batches     : "
        f"{measured_batches}"
    )

    print(
        f"Measured triplets    : "
        f"{measured_triplets}"
    )

    print(
        f"Total GPU time       : "
        f"{total_gpu_time_ms:.3f} ms"
    )

    print(
        f"Mean time / batch    : "
        f"{ms_per_batch:.3f} ms"
    )

    print(
        f"Mean time / triplet  : "
        f"{ms_per_triplet:.3f} ms"
    )

    print(
        f"Throughput           : "
        f"{throughput:.2f} triplets/s"
    )

    print(
        f"Peak inference VRAM  : "
        f"{peak_vram_mb:.2f} MB"
    )

    print(
        f"Parameters           : "
        f"{total_params:,}"
    )

    if args.test_2afc is not None:

        print(
            f"Test 2AFC            : "
            f"{args.test_2afc:.2f}%"
        )

    # =================================================================
    # SAVE
    # =================================================================

    append_result(
        args.output,
        result,
    )

    print()
    print(
        f"Result appended to: "
        f"{args.output}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()