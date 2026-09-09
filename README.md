# Efficient Attention Impact on Transformer Backbones for Perceptual Similarity

This project investigates the impact of efficient attention mechanisms on Vision Transformer backbones for perceptual image similarity.

The project is based on a **DINO ViT-B/16** backbone and follows a DreamSim-like perceptual similarity pipeline trained and evaluated on the **NIGHTS (Novel Image Generations with Human-Tested Similarities)** dataset.

The main goal is to compare standard self-attention against different efficient alternatives in terms of:

- perceptual similarity performance (2AFC accuracy);
- inference latency;
- inference throughput;
- GPU memory usage;
- model size;
- training cost.

Three efficient mechanisms are implemented:

- **Spatial Reduction Attention (SRA)**, inspired by Pyramid Vision Transformer;
- **MetaFormer / PoolFormer-style token mixing**;
- **Mixture-of-Heads Attention (MoH)**.

Each efficient mechanism can be applied either to:

- **Full configuration**: all 12 Transformer blocks;
- **Half configuration**: only the last 6 Transformer blocks.

---

## Project Structure

```text
ComputerVision_Project/
│
├── data/
│   └── nights/
│       ├── data.csv
│       ├── ref/
│       └── distort/
│
├── src/
│   ├── datasets/
│   │   └── nights.py
│   │
│   └── models/
│       ├── attention/
│       │   ├── spatial_reduction.py
│       │   ├── metaformer_pooling.py
│       │   └── mixture_of_heads.py
│       │
│       ├── transforms.py
│       ├── perceptual_similarity.py
│       ├── vit_baseline.py
│       ├── vit_efficient.py
│       ├── vit_metaformer.py
│       └── vit_moh.py
│
├── scripts/
│   ├── train_baseline.py
│   ├── train_sra.py
│   ├── train_metaformer.py
│   ├── train_moh.py
│   │
│   ├── evaluate_baseline.py
│   ├── evaluate_sra.py
│   ├── evaluate_metaformer.py
│   ├── evaluate_moh.py
│   │
│   └── benchmark_models.py
│
├── checkpoints/
├── results/
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository and enter the project directory:

```bash
git clone https://github.com/SasakiKun22/ComputerVision_Project.git
cd ComputerVision_Project
```

It is recommended to use a dedicated Python environment.

For example, using Conda:

```bash
conda create -n cv-efficient-attention python=3.10
conda activate cv-efficient-attention
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

A CUDA-compatible PyTorch installation is strongly recommended for training and benchmarking.

You can verify CUDA availability with:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

# Dataset

## NIGHTS Dataset

This project uses the **NIGHTS (Novel Image Generations with Human-Tested Similarities)** dataset introduced with DreamSim.

Each sample contains:

- a reference image;
- two distorted/generated images;
- human similarity judgments indicating which image is perceptually closer to the reference.

The dataset is filtered using samples with at least 6 votes.

The expected directory structure is:

```text
data/
└── nights/
    ├── data.csv
    ├── ref/
    └── distort/
```

## Download

The dataset can be downloaded from the official DreamSim / NIGHTS release:

**NIGHTS Dataset:**  
(https://data.csail.mit.edu/nights/nights.zip)

After downloading and extracting the dataset, place it inside:

```text
./data/
```

The final structure must therefore be:

```text
./data/nights/data.csv
./data/nights/ref/
./data/nights/distort/
```

The dataset itself is not included in this repository because of its size.

---

# Pretrained and Fine-tuned Weights

The fine-tuned model checkpoints are not stored directly in the Git repository because of their size.

They can be downloaded from:

**Trained Model Weights:**  
`https://drive.google.com/drive/folders/10eEp2_Zlf1QLVPi16RaevBg6HGSZS8lU?usp=drive_link`

After downloading them, place all checkpoints inside:

```text
./checkpoints/
```

The expected checkpoint names are:

```text
checkpoints/
├── baseline_best.pth
├── sra_all_sr2_best.pth
├── sra_last_half_sr2_best.pth
├── metaformer_all_pool3_best.pth
├── metaformer_last_half_pool3_best.pth
├── moh_all_shared2_routed4_best.pth
└── moh_last_half_shared2_routed4_best.pth
```

The original DINO ViT-B/16 pretrained weights are automatically loaded when the models are initialized.

---

# Models

## DINO Baseline

The baseline uses the standard pretrained **DINO ViT-B/16** architecture with 12 standard multi-head self-attention Transformer blocks.

```text
12 × Standard Self-Attention
```

---

## Spatial Reduction Attention (SRA)

The SRA implementation is inspired by Pyramid Vision Transformer.

Queries preserve the original sequence resolution, while Key and Value tokens are spatially reduced.

The experiments use:

```text
sr_ratio = 2
```

### SRA Full

```text
Blocks 0-11 → SRA
```

### SRA Half

```text
Blocks 0-5  → Standard Attention
Blocks 6-11 → SRA
```

---

## MetaFormer / PoolFormer

This variant replaces self-attention with a PoolFormer-style pooling token mixer.

The patch tokens are mixed using local average pooling.

The experiments use:

```text
pool_size = 3
```

### MetaFormer Full

```text
Blocks 0-11 → Pooling Token Mixer
```

### MetaFormer Half

```text
Blocks 0-5  → Standard Attention
Blocks 6-11 → Pooling Token Mixer
```

---

## Mixture-of-Heads Attention (MoH)

MoH dynamically routes attention-head contributions.

The implemented configuration uses:

```text
12 total attention heads
2 shared heads
4 routed heads
6 active heads per token
```

### MoH Full

```text
Blocks 0-11 → Mixture-of-Heads Attention
```

### MoH Half

```text
Blocks 0-5  → Standard Attention
Blocks 6-11 → Mixture-of-Heads Attention
```

---

# Training

All models use the same main training protocol to provide a fair comparison.

Default configuration:

```text
Epochs                  5
Train subset            50%
Validation subset       50%
Batch size              4
Gradient accumulation   4
Effective batch size    16
Learning rate           1e-5
Seed                    42
```

The best checkpoint is selected according to **validation 2AFC accuracy**.

Training curves and history files are automatically stored inside:

```text
./results/
```

---

## Train Baseline

```bash
python -m scripts.train_baseline \
    --epochs 5 \
    --train-fraction 0.5 \
    --val-fraction 0.5 \
    --batch-size 4 \
    --grad-accum 4 \
    --lr 1e-5
```

---

## Train SRA Full

```bash
python -m scripts.train_sra \
    --mode all \
    --sr-ratio 2 \
    --epochs 5 \
    --train-fraction 0.5 \
    --val-fraction 0.5 \
    --batch-size 4 \
    --grad-accum 4 \
    --lr 1e-5
```

## Train SRA Half

```bash
python -m scripts.train_sra \
    --mode last_half \
    --sr-ratio 2 \
    --epochs 5 \
    --train-fraction 0.5 \
    --val-fraction 0.5 \
    --batch-size 4 \
    --grad-accum 4 \
    --lr 1e-5
```

---

## Train MetaFormer Full

```bash
python -m scripts.train_metaformer \
    --mode all \
    --pool-size 3 \
    --epochs 5 \
    --train-fraction 0.5 \
    --val-fraction 0.5 \
    --batch-size 4 \
    --grad-accum 4 \
    --lr 1e-5
```

## Train MetaFormer Half

```bash
python -m scripts.train_metaformer \
    --mode last_half \
    --pool-size 3 \
    --epochs 5 \
    --train-fraction 0.5 \
    --val-fraction 0.5 \
    --batch-size 4 \
    --grad-accum 4 \
    --lr 1e-5
```

---

## Train MoH Full

```bash
python -m scripts.train_moh \
    --mode all \
    --shared-heads 2 \
    --routed-heads 4 \
    --aux-lambda 0.01 \
    --epochs 5 \
    --train-fraction 0.5 \
    --val-fraction 0.5 \
    --batch-size 4 \
    --grad-accum 4 \
    --lr 1e-5
```

## Train MoH Half

```bash
python -m scripts.train_moh \
    --mode last_half \
    --shared-heads 2 \
    --routed-heads 4 \
    --aux-lambda 0.01 \
    --epochs 5 \
    --train-fraction 0.5 \
    --val-fraction 0.5 \
    --batch-size 4 \
    --grad-accum 4 \
    --lr 1e-5
```

---

# Evaluation

Performance is evaluated using **2-Alternative Forced Choice (2AFC)** accuracy on the full NIGHTS test split.

Higher values indicate better agreement with human perceptual judgments.

---

## Baseline

```bash
python -m scripts.evaluate_baseline \
    --checkpoint ./checkpoints/baseline_best.pth \
    --subset-fraction 1.0 \
    --batch-size 32
```

---

## SRA Full

```bash
python -m scripts.evaluate_sra \
    --checkpoint ./checkpoints/sra_all_sr2_best.pth \
    --subset-fraction 1.0 \
    --batch-size 32
```

## SRA Half

```bash
python -m scripts.evaluate_sra \
    --checkpoint ./checkpoints/sra_last_half_sr2_best.pth \
    --subset-fraction 1.0 \
    --batch-size 32
```

---

## MetaFormer Full

```bash
python -m scripts.evaluate_metaformer \
    --checkpoint ./checkpoints/metaformer_all_pool3_best.pth \
    --subset-fraction 1.0 \
    --batch-size 32
```

## MetaFormer Half

```bash
python -m scripts.evaluate_metaformer \
    --checkpoint ./checkpoints/metaformer_last_half_pool3_best.pth \
    --subset-fraction 1.0 \
    --batch-size 32
```

---

## MoH Full

```bash
python -m scripts.evaluate_moh \
    --checkpoint ./checkpoints/moh_all_shared2_routed4_best.pth \
    --subset-fraction 1.0 \
    --batch-size 32
```

## MoH Half

```bash
python -m scripts.evaluate_moh \
    --checkpoint ./checkpoints/moh_last_half_shared2_routed4_best.pth \
    --subset-fraction 1.0 \
    --batch-size 32
```

---

# GPU Benchmark

The benchmark script automatically performs:

1. full NIGHTS test-set 2AFC evaluation;
2. GPU warm-up;
3. inference latency measurement;
4. throughput measurement;
5. peak inference VRAM measurement;
6. parameter counting.

Therefore, the 2AFC value does **not** need to be manually provided.

The benchmark reports:

```text
Test 2AFC
ms / batch
ms / triplet
triplets / second
ms / image
images / second
peak inference VRAM
number of parameters
```

Results are stored in:

```text
results/benchmark/inference_benchmark.csv
```

---

## Benchmark Baseline

```bash
python -m scripts.benchmark_models \
    --model baseline \
    --checkpoint ./checkpoints/baseline_best.pth \
    --batch-size 32
```

## Benchmark SRA Full

```bash
python -m scripts.benchmark_models \
    --model sra_all \
    --checkpoint ./checkpoints/sra_all_sr2_best.pth \
    --batch-size 32
```

## Benchmark SRA Half

```bash
python -m scripts.benchmark_models \
    --model sra_half \
    --checkpoint ./checkpoints/sra_last_half_sr2_best.pth \
    --batch-size 32
```

## Benchmark MetaFormer Full

```bash
python -m scripts.benchmark_models \
    --model metaformer_all \
    --checkpoint ./checkpoints/metaformer_all_pool3_best.pth \
    --batch-size 32
```

## Benchmark MetaFormer Half

```bash
python -m scripts.benchmark_models \
    --model metaformer_half \
    --checkpoint ./checkpoints/metaformer_last_half_pool3_best.pth \
    --batch-size 32
```

## Benchmark MoH Full

```bash
python -m scripts.benchmark_models \
    --model moh_all \
    --checkpoint ./checkpoints/moh_all_shared2_routed4_best.pth \
    --batch-size 32
```

## Benchmark MoH Half

```bash
python -m scripts.benchmark_models \
    --model moh_half \
    --checkpoint ./checkpoints/moh_last_half_shared2_routed4_best.pth \
    --batch-size 32
```

---

# Run All Benchmarks

To benchmark every architecture sequentially:

```bash
rm -f results/benchmark/inference_benchmark.csv && \
python -m scripts.benchmark_models \
    --model baseline \
    --checkpoint ./checkpoints/baseline_best.pth \
    --batch-size 32 && \
python -m scripts.benchmark_models \
    --model sra_all \
    --checkpoint ./checkpoints/sra_all_sr2_best.pth \
    --batch-size 32 && \
python -m scripts.benchmark_models \
    --model sra_half \
    --checkpoint ./checkpoints/sra_last_half_sr2_best.pth \
    --batch-size 32 && \
python -m scripts.benchmark_models \
    --model metaformer_all \
    --checkpoint ./checkpoints/metaformer_all_pool3_best.pth \
    --batch-size 32 && \
python -m scripts.benchmark_models \
    --model metaformer_half \
    --checkpoint ./checkpoints/metaformer_last_half_pool3_best.pth \
    --batch-size 32 && \
python -m scripts.benchmark_models \
    --model moh_all \
    --checkpoint ./checkpoints/moh_all_shared2_routed4_best.pth \
    --batch-size 32 && \
python -m scripts.benchmark_models \
    --model moh_half \
    --checkpoint ./checkpoints/moh_last_half_shared2_routed4_best.pth \
    --batch-size 32
```

The resulting CSV can be inspected with:

```bash
cat results/benchmark/inference_benchmark.csv
```

---

# Experimental Variants

The final comparison contains seven trained architectures:

| Model | Blocks 0-5 | Blocks 6-11 |
|---|---|---|
| DINO Baseline | Standard Attention | Standard Attention |
| SRA Full | SRA | SRA |
| SRA Half | Standard Attention | SRA |
| MetaFormer Full | Pooling Mixer | Pooling Mixer |
| MetaFormer Half | Standard Attention | Pooling Mixer |
| MoH Full | MoH | MoH |
| MoH Half | Standard Attention | MoH |

This setup allows both:

- comparison between different efficient attention mechanisms;
- ablation between full and partial attention replacement.

---

# Notes

- The dataset is not committed to Git.
- Model checkpoints are not committed to Git.
- Training and validation subsets are sampled deterministically using seed `42`.
- Final evaluation always uses the complete filtered NIGHTS test split.
- The benchmark should be executed under identical hardware and batch-size conditions for all models.
- GPU timing excludes data-loading time and measures the model forward pass.
- MoH uses dense PyTorch attention operations with dynamic head routing. Therefore, theoretical head sparsity does not necessarily translate directly into proportional wall-clock speedup without specialized sparse CUDA kernels.

---

# References

This project builds upon the following works:

- DreamSim: *Learning New Dimensions of Human Visual Similarity using Synthetic Data*
- DINO: *Emerging Properties in Self-Supervised Vision Transformers*
- Pyramid Vision Transformer
- MetaFormer / PoolFormer
- Mixture-of-Head Attention

---

# Authors

Computer Vision Project

**Author:** `Mattia Di Marco, Simone La Bella`

**Course:** Computer Vision

**Academic Year:** `2025/2026`
