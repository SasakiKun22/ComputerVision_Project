import argparse
import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.nights import NIGHTSDataset
from src.models.transforms import get_dino_transform
from src.models.vit_baseline import DINOViTBaseline
from src.models.vit_efficient import DINOViTEfficient
from src.models.perceptual_similarity import PerceptualSimilarityModel


# =====================================================================
# ARGUMENTS
# =====================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Benchmark NIGHTS perceptual similarity models."
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
        default=50,
    )

    parser.add_argument(
        "--sr-ratio",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--amp",
        action="store_true",
        help="Benchmark using FP16 autocast.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./results/benchmark/inference_benchmark.csv",
    )

    return parser.parse_args()


# =====================================================================
# MODEL
# =====================================================================

def build_model(args, device):

    if args.model == "baseline":

        encoder = DINOViTBaseline(
            pretrained=True,
            freeze=False,
        )

    elif args.model == "sra_all":

        encoder = DINOViTEfficient(
            pretrained=True,
            freeze=False,
            sr_ratio=args.sr_ratio,
            efficient_layers=list(range(12)),
        )

    elif args.model == "sra_half":

        encoder = DINOViTEfficient(
            pretrained=True,
            freeze=False,
            sr_ratio=args.sr_ratio,
            efficient_layers=list(range(6, 12)),
        )

    else:

        raise ValueError(
            f"Unsupported model: {args.model}"
        )

    model = PerceptualSimilarityModel(
        encoder
    ).to(device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


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
# SAVE CSV
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
        "batch_size",
        "amp",
        "num_triplets",
        "total_time_s",
        "ms_per_triplet",
        "triplets_per_second",
        "peak_vram_mb",
        "parameters",
        "trainable_parameters",
    ]

    with open(
        output_path,
        "a",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            result
        )


# =====================================================================
# MAIN
# =====================================================================

def main():

    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this benchmark."
        )

    device = torch.device(
        "cuda"
    )

    print("=" * 72)
    print("GPU INFERENCE BENCHMARK")
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

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # =================================================================
    # MODEL
    # =================================================================

    print()
    print("Loading model...")

    model = build_model(
        args,
        device,
    )

    total_params, trainable_params = (
        count_parameters(model)
    )

    print(
        f"Parameters: "
        f"{total_params:,}"
    )

    # =================================================================
    # PRELOAD BATCHES
    #
    # We load batches before timing so that disk/DataLoader speed does
    # not contaminate the GPU inference benchmark.
    # =================================================================

    required_batches = (
        args.warmup_batches
        + args.benchmark_batches
    )

    batches = []

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

        batches.append(
            (
                reference,
                left,
                right,
            )
        )

        if len(batches) >= required_batches:
            break

    if len(batches) <= args.warmup_batches:

        raise RuntimeError(
            "Not enough batches available for benchmark."
        )

    actual_benchmark_batches = min(
        args.benchmark_batches,
        len(batches)
        - args.warmup_batches,
    )

    # =================================================================
    # WARMUP
    # =================================================================

    print()
    print(
        f"Warm-up: "
        f"{args.warmup_batches} batches"
    )

    with torch.inference_mode():

        for i in range(
            args.warmup_batches
        ):

            reference, left, right = (
                batches[i]
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

    # =================================================================
    # RESET MEMORY STATISTICS
    # =================================================================

    torch.cuda.reset_peak_memory_stats(
        device
    )

    # =================================================================
    # BENCHMARK
    # =================================================================

    print(
        f"Benchmark: "
        f"{actual_benchmark_batches} batches"
    )

    total_triplets = (
        actual_benchmark_batches
        * args.batch_size
    )

    torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.inference_mode():

        start_index = (
            args.warmup_batches
        )

        end_index = (
            start_index
            + actual_benchmark_batches
        )

        for i in range(
            start_index,
            end_index,
        ):

            reference, left, right = (
                batches[i]
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

    end_time = time.perf_counter()

    # =================================================================
    # RESULTS
    # =================================================================

    total_time = (
        end_time
        - start_time
    )

    ms_per_triplet = (
        total_time
        / total_triplets
        * 1000
    )

    throughput = (
        total_triplets
        / total_time
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

        "batch_size":
            args.batch_size,

        "amp":
            args.amp,

        "num_triplets":
            total_triplets,

        "total_time_s":
            total_time,

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
    }

    print()
    print("=" * 72)
    print("RESULTS")
    print("=" * 72)

    print(
        f"Measured triplets   : "
        f"{total_triplets}"
    )

    print(
        f"Total inference time: "
        f"{total_time:.4f} s"
    )

    print(
        f"Latency             : "
        f"{ms_per_triplet:.3f} ms/triplet"
    )

    print(
        f"Throughput          : "
        f"{throughput:.2f} triplets/s"
    )

    print(
        f"Peak VRAM           : "
        f"{peak_vram_mb:.2f} MB"
    )

    print(
        f"Parameters          : "
        f"{total_params:,}"
    )

    print()

    append_result(
        args.output,
        result,
    )

    print(
        f"Saved to: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()