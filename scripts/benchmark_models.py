import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.nights import NIGHTSDataset
from src.models.transforms import get_dino_transform

from src.models.vit_baseline import DINOViTBaseline
from src.models.vit_efficient import DINOViTEfficient
from src.models.vit_metaformer import DINOViTMetaFormer
from src.models.vit_moh import DINOViTMoH

from src.models.perceptual_similarity import (
    PerceptualSimilarityModel,
)


# =====================================================================
# ARGUMENTS
# =====================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate NIGHTS 2AFC and benchmark GPU inference "
            "for all perceptual similarity models."
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
            "moh_all",
            "moh_half",
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

    parser.add_argument(
        "--amp",
        action="store_true",
        help=(
            "Use FP16 autocast during the GPU speed benchmark. "
            "2AFC evaluation always remains FP32."
        ),
    )

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
# CHECKPOINT UTILITIES
# =====================================================================

def require_checkpoint_keys(
    checkpoint,
    keys,
    model_name,
):

    missing = [
        key
        for key in keys
        if key not in checkpoint
    ]

    if missing:

        raise ValueError(
            f"Checkpoint for '{model_name}' is missing "
            f"required metadata: {missing}. "
            f"Make sure it was produced with the updated "
            f"training script."
        )


def validate_mode(
    requested_model,
    checkpoint_mode,
):

    expected_modes = {
        "sra_all": "all",
        "sra_half": "last_half",
        "metaformer_all": "all",
        "metaformer_half": "last_half",
        "moh_all": "all",
        "moh_half": "last_half",
    }

    if requested_model == "baseline":
        return

    expected_mode = expected_modes[
        requested_model
    ]

    if checkpoint_mode != expected_mode:

        raise ValueError(
            "\nModel/checkpoint mismatch.\n"
            f"Requested model : {requested_model}\n"
            f"Expected mode   : {expected_mode}\n"
            f"Checkpoint mode : {checkpoint_mode}\n\n"
            "Use the checkpoint corresponding to the "
            "requested architecture."
        )


def validate_layers(
    mode,
    layers,
    layer_name,
):

    if mode == "all":

        expected_layers = list(
            range(12)
        )

    elif mode == "last_half":

        expected_layers = list(
            range(6, 12)
        )

    else:

        raise ValueError(
            f"Unsupported checkpoint mode: {mode}"
        )

    if list(layers) != expected_layers:

        raise ValueError(
            f"Unexpected {layer_name} configuration.\n"
            f"Mode            : {mode}\n"
            f"Expected layers : {expected_layers}\n"
            f"Checkpoint      : {list(layers)}"
        )


# =====================================================================
# BUILD MODEL
# =====================================================================

def build_model(
    args,
    device,
):

    print(
        "Reading checkpoint metadata..."
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
    )

    # =================================================================
    # BASELINE
    # =================================================================

    if args.model == "baseline":

        encoder = DINOViTBaseline(
            pretrained=True,
            freeze=False,
        )

        architecture_info = {
            "architecture":
                "Standard Attention",

            "mode":
                "baseline",

            "efficient_layers":
                0,

            "configuration":
                "standard",

            "active_heads":
                "12/12",
        }

    # =================================================================
    # SRA
    # =================================================================

    elif args.model in [
        "sra_all",
        "sra_half",
    ]:

        require_checkpoint_keys(
            checkpoint,
            [
                "mode",
                "sr_ratio",
                "efficient_layers",
            ],
            args.model,
        )

        mode = checkpoint[
            "mode"
        ]

        sr_ratio = checkpoint[
            "sr_ratio"
        ]

        efficient_layers = checkpoint[
            "efficient_layers"
        ]

        validate_mode(
            args.model,
            mode,
        )

        validate_layers(
            mode,
            efficient_layers,
            "SRA layers",
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

            "mode":
                mode,

            "efficient_layers":
                len(efficient_layers),

            "configuration":
                f"sr_ratio={sr_ratio}",

            "active_heads":
                "12/12",
        }

    # =================================================================
    # METAFORMER
    # =================================================================

    elif args.model in [
        "metaformer_all",
        "metaformer_half",
    ]:

        require_checkpoint_keys(
            checkpoint,
            [
                "mode",
                "pool_size",
                "metaformer_layers",
            ],
            args.model,
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

        validate_mode(
            args.model,
            mode,
        )

        validate_layers(
            mode,
            metaformer_layers,
            "MetaFormer layers",
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
                "MetaFormer Pooling",

            "mode":
                mode,

            "efficient_layers":
                len(metaformer_layers),

            "configuration":
                f"pool_size={pool_size}",

            "active_heads":
                "N/A",
        }

    # =================================================================
    # MIXTURE-OF-HEADS
    # =================================================================

    elif args.model in [
        "moh_all",
        "moh_half",
    ]:

        require_checkpoint_keys(
            checkpoint,
            [
                "mode",
                "shared_heads",
                "routed_heads",
                "moh_layers",
            ],
            args.model,
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

        validate_mode(
            args.model,
            mode,
        )

        validate_layers(
            mode,
            moh_layers,
            "MoH layers",
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

        architecture_info = {
            "architecture":
                "Mixture-of-Heads",

            "mode":
                mode,

            "efficient_layers":
                len(moh_layers),

            "configuration":
                (
                    f"shared={shared_heads},"
                    f"routed={routed_heads}"
                ),

            "active_heads":
                (
                    f"{shared_heads + routed_heads}/12"
                ),
        }

    else:

        raise ValueError(
            f"Unsupported model: "
            f"{args.model}"
        )

    # =================================================================
    # DREAMSIM-LIKE WRAPPER
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

    return (
        model,
        architecture_info,
        checkpoint,
    )


# =====================================================================
# PREPARE GPU BATCH
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
# FULL TEST 2AFC
# =====================================================================

@torch.inference_mode()
def evaluate_2afc(
    model,
    loader,
    device,
):

    model.eval()

    correct = 0
    total = 0

    distance_left_sum = 0.0
    distance_right_sum = 0.0

    print()
    print("=" * 72)
    print(
        "FULL TEST 2AFC EVALUATION"
    )
    print("=" * 72)

    for batch_idx, batch in enumerate(
        loader
    ):

        (
            reference,
            left,
            right,
        ) = prepare_batch(
            batch,
            device,
        )

        target = batch[
            "target"
        ].long().to(
            device,
            non_blocking=True,
        )

        # -------------------------------------------------------------
        # FP32 evaluation.
        #
        # No autocast here, consistent with the standalone evaluator
        # scripts.
        # -------------------------------------------------------------

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
            (batch_idx + 1)
            % 10
            == 0
            or batch_idx
            == len(loader) - 1
        ):

            running_accuracy = (
                correct
                / total
                * 100.0
            )

            print(
                f"Batch "
                f"{batch_idx + 1:4d}/"
                f"{len(loader):4d}"
                f" | "
                f"2AFC: "
                f"{running_accuracy:.2f}%"
            )

    accuracy = (
        correct
        / total
    )

    mean_distance_left = (
        distance_left_sum
        / total
    )

    mean_distance_right = (
        distance_right_sum
        / total
    )

    print()

    print(
        f"Correct    : "
        f"{correct}/{total}"
    )

    print(
        f"Test 2AFC  : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Mean d(L)  : "
        f"{mean_distance_left:.6f}"
    )

    print(
        f"Mean d(R)  : "
        f"{mean_distance_right:.6f}"
    )

    return {
        "accuracy":
            accuracy,

        "correct":
            correct,

        "total":
            total,

        "mean_distance_left":
            mean_distance_left,

        "mean_distance_right":
            mean_distance_right,
    }


# =====================================================================
# GPU BENCHMARK
# =====================================================================

@torch.inference_mode()
def benchmark_inference(
    model,
    loader,
    device,
    warmup_batches,
    benchmark_batches,
    use_amp,
):

    model.eval()

    print()
    print("=" * 72)
    print(
        "GPU INFERENCE BENCHMARK"
    )
    print("=" * 72)

    loader_iterator = iter(
        loader
    )

    # =================================================================
    # WARM-UP
    # =================================================================

    print(
        f"Warm-up batches: "
        f"{warmup_batches}"
    )

    for _ in range(
        warmup_batches
    ):

        try:

            batch = next(
                loader_iterator
            )

        except StopIteration:

            raise RuntimeError(
                "Not enough batches "
                "for the requested warm-up."
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
            enabled=use_amp,
        ):

            model(
                reference,
                left,
                right,
            )

    torch.cuda.synchronize()

    # -----------------------------------------------------------------
    # Remove final warm-up references before resetting peak memory.
    # -----------------------------------------------------------------

    del batch
    del reference
    del left
    del right

    torch.cuda.synchronize()

    # =================================================================
    # MEMORY RESET
    # =================================================================

    torch.cuda.reset_peak_memory_stats(
        device
    )

    # =================================================================
    # TIMED BENCHMARK
    # =================================================================

    total_gpu_time_ms = 0.0

    measured_batches = 0
    measured_triplets = 0

    print(
        f"Requested benchmark batches: "
        f"{benchmark_batches}"
    )

    for _ in range(
        benchmark_batches
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

        # -------------------------------------------------------------
        # Ensure host -> GPU transfer is completed before timing.
        # -------------------------------------------------------------

        torch.cuda.synchronize()

        start_event = torch.cuda.Event(
            enable_timing=True
        )

        end_event = torch.cuda.Event(
            enable_timing=True
        )

        start_event.record()

        # -------------------------------------------------------------
        # Model forward only.
        # -------------------------------------------------------------

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp,
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
            "No benchmark batches "
            "were measured."
        )

    # =================================================================
    # METRICS
    # =================================================================

    total_gpu_time_seconds = (
        total_gpu_time_ms
        / 1000.0
    )

    ms_per_batch = (
        total_gpu_time_ms
        / measured_batches
    )

    ms_per_triplet = (
        total_gpu_time_ms
        / measured_triplets
    )

    triplets_per_second = (
        measured_triplets
        / total_gpu_time_seconds
    )

    # Each triplet contains:
    #
    # reference + left + right
    #
    measured_images = (
        measured_triplets
        * 3
    )

    ms_per_image = (
        total_gpu_time_ms
        / measured_images
    )

    images_per_second = (
        measured_images
        / total_gpu_time_seconds
    )

    peak_vram_mb = (
        torch.cuda.max_memory_allocated(
            device
        )
        / (1024 ** 2)
    )

    print()

    print(
        f"Measured batches    : "
        f"{measured_batches}"
    )

    print(
        f"Measured triplets   : "
        f"{measured_triplets}"
    )

    print(
        f"Total GPU time      : "
        f"{total_gpu_time_ms:.3f} ms"
    )

    print(
        f"Time / batch        : "
        f"{ms_per_batch:.3f} ms"
    )

    print(
        f"Time / triplet      : "
        f"{ms_per_triplet:.3f} ms"
    )

    print(
        f"Triplets / second   : "
        f"{triplets_per_second:.2f}"
    )

    print(
        f"Time / image        : "
        f"{ms_per_image:.3f} ms"
    )

    print(
        f"Images / second     : "
        f"{images_per_second:.2f}"
    )

    print(
        f"Peak inference VRAM : "
        f"{peak_vram_mb:.2f} MB"
    )

    return {
        "benchmark_batches":
            measured_batches,

        "num_triplets":
            measured_triplets,

        "num_images":
            measured_images,

        "total_gpu_time_ms":
            total_gpu_time_ms,

        "ms_per_batch":
            ms_per_batch,

        "ms_per_triplet":
            ms_per_triplet,

        "triplets_per_second":
            triplets_per_second,

        "ms_per_image":
            ms_per_image,

        "images_per_second":
            images_per_second,

        "peak_vram_mb":
            peak_vram_mb,
    }


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
        "architecture",
        "mode",
        "efficient_layers",
        "configuration",
        "active_heads",

        "batch_size",
        "benchmark_amp",

        "test_correct",
        "test_total",
        "test_2afc",

        "mean_distance_left",
        "mean_distance_right",

        "benchmark_batches",
        "num_triplets",
        "num_images",

        "total_gpu_time_ms",
        "ms_per_batch",
        "ms_per_triplet",
        "triplets_per_second",

        "ms_per_image",
        "images_per_second",

        "peak_vram_mb",

        "parameters",
        "trainable_parameters",

        "checkpoint_epoch",
        "checkpoint_val_2afc",
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
# MAIN
# =====================================================================

def main():

    args = parse_args()

    # =================================================================
    # CUDA
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
        "NIGHTS MODEL EVALUATION + BENCHMARK"
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
        "Benchmark AMP:",
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

    # -----------------------------------------------------------------
    # Evaluation uses every test sample.
    # -----------------------------------------------------------------

    evaluation_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # -----------------------------------------------------------------
    # Benchmark only uses complete batches so every timed batch has
    # identical dimensions.
    # -----------------------------------------------------------------

    benchmark_loader = DataLoader(
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
        "Evaluation batches:",
        len(evaluation_loader),
    )

    print(
        "Benchmark batches available:",
        len(benchmark_loader),
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

    (
        total_params,
        trainable_params,
    ) = count_parameters(
        model
    )

    print()

    print(
        "Architecture:",
        architecture_info[
            "architecture"
        ],
    )

    print(
        "Mode:",
        architecture_info[
            "mode"
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
            "configuration"
        ],
    )

    print(
        "Active heads:",
        architecture_info[
            "active_heads"
        ],
    )

    print(
        f"Parameters: "
        f"{total_params:,}"
    )

    # =================================================================
    # FULL TEST EVALUATION
    # =================================================================

    evaluation_results = (
        evaluate_2afc(
            model=model,
            loader=evaluation_loader,
            device=device,
        )
    )

    torch.cuda.synchronize()

    # =================================================================
    # GPU BENCHMARK
    # =================================================================

    benchmark_results = (
        benchmark_inference(
            model=model,
            loader=benchmark_loader,
            device=device,
            warmup_batches=(
                args.warmup_batches
            ),
            benchmark_batches=(
                args.benchmark_batches
            ),
            use_amp=args.amp,
        )
    )

    # =================================================================
    # CHECKPOINT METADATA
    # =================================================================

    checkpoint_epoch = (
        checkpoint.get(
            "epoch",
            ""
        )
    )

    checkpoint_val_accuracy = (
        checkpoint.get(
            "val_accuracy",
            None,
        )
    )

    if (
        checkpoint_val_accuracy
        is not None
    ):

        checkpoint_val_2afc = (
            checkpoint_val_accuracy
            * 100.0
        )

    else:

        checkpoint_val_2afc = ""

    # =================================================================
    # RESULT ROW
    # =================================================================

    result = {

        "model":
            args.model,

        "architecture":
            architecture_info[
                "architecture"
            ],

        "mode":
            architecture_info[
                "mode"
            ],

        "efficient_layers":
            architecture_info[
                "efficient_layers"
            ],

        "configuration":
            architecture_info[
                "configuration"
            ],

        "active_heads":
            architecture_info[
                "active_heads"
            ],

        "batch_size":
            args.batch_size,

        "benchmark_amp":
            args.amp,

        # -------------------------------------------------------------
        # Test metrics
        # -------------------------------------------------------------

        "test_correct":
            evaluation_results[
                "correct"
            ],

        "test_total":
            evaluation_results[
                "total"
            ],

        "test_2afc":
            evaluation_results[
                "accuracy"
            ]
            * 100.0,

        "mean_distance_left":
            evaluation_results[
                "mean_distance_left"
            ],

        "mean_distance_right":
            evaluation_results[
                "mean_distance_right"
            ],

        # -------------------------------------------------------------
        # Benchmark metrics
        # -------------------------------------------------------------

        "benchmark_batches":
            benchmark_results[
                "benchmark_batches"
            ],

        "num_triplets":
            benchmark_results[
                "num_triplets"
            ],

        "num_images":
            benchmark_results[
                "num_images"
            ],

        "total_gpu_time_ms":
            benchmark_results[
                "total_gpu_time_ms"
            ],

        "ms_per_batch":
            benchmark_results[
                "ms_per_batch"
            ],

        "ms_per_triplet":
            benchmark_results[
                "ms_per_triplet"
            ],

        "triplets_per_second":
            benchmark_results[
                "triplets_per_second"
            ],

        "ms_per_image":
            benchmark_results[
                "ms_per_image"
            ],

        "images_per_second":
            benchmark_results[
                "images_per_second"
            ],

        "peak_vram_mb":
            benchmark_results[
                "peak_vram_mb"
            ],

        # -------------------------------------------------------------
        # Model size
        # -------------------------------------------------------------

        "parameters":
            total_params,

        "trainable_parameters":
            trainable_params,

        # -------------------------------------------------------------
        # Checkpoint information
        # -------------------------------------------------------------

        "checkpoint_epoch":
            checkpoint_epoch,

        "checkpoint_val_2afc":
            checkpoint_val_2afc,
    }

    # =================================================================
    # SAVE
    # =================================================================

    append_result(
        args.output,
        result,
    )

    # =================================================================
    # SUMMARY
    # =================================================================

    print()
    print("=" * 72)
    print(
        "FINAL SUMMARY"
    )
    print("=" * 72)

    print(
        f"Model                : "
        f"{args.model}"
    )

    print(
        f"Architecture         : "
        f"{architecture_info['architecture']}"
    )

    print(
        f"Mode                 : "
        f"{architecture_info['mode']}"
    )

    print(
        f"Test 2AFC            : "
        f"{result['test_2afc']:.2f}%"
    )

    print(
        f"Latency              : "
        f"{result['ms_per_triplet']:.3f} "
        f"ms/triplet"
    )

    print(
        f"Throughput           : "
        f"{result['triplets_per_second']:.2f} "
        f"triplets/s"
    )

    print(
        f"Peak inference VRAM  : "
        f"{result['peak_vram_mb']:.2f} MB"
    )

    print(
        f"Parameters           : "
        f"{total_params:,}"
    )

    print()

    print(
        f"Result saved to: "
        f"{args.output}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()