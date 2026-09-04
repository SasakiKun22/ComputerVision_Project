import pandas as pd
import matplotlib.pyplot as plt


# =====================================================================
# CONFIG
# =====================================================================

CSV_PATH = (
    "./results/benchmark/"
    "inference_benchmark.csv"
)

OUTPUT_PNG = (
    "./results/benchmark/"
    "latency_vs_2afc.png"
)

OUTPUT_PDF = (
    "./results/benchmark/"
    "latency_vs_2afc.pdf"
)

MAIN_COLOR = "#822433"

BASELINE_COLOR = "#666666"


# =====================================================================
# LOAD DATA
# =====================================================================

df = pd.read_csv(
    CSV_PATH
)


# ---------------------------------------------------------------------
# If the CSV contains multiple benchmark runs for the same model,
# keep only the most recent one.
# ---------------------------------------------------------------------

df = df.drop_duplicates(
    subset="model",
    keep="last",
)


# =====================================================================
# MODEL GROUPS
# =====================================================================

families = {
    "SRA": {
        "full": "sra_all",
        "half": "sra_half",
    },

    "MetaFormer": {
        "full": "metaformer_all",
        "half": "metaformer_half",
    },

    "MoH": {
        "full": "moh_all",
        "half": "moh_half",
    },
}


# =====================================================================
# CHECK REQUIRED MODELS
# =====================================================================

required_models = [
    "baseline",
    "sra_all",
    "sra_half",
    "metaformer_all",
    "metaformer_half",
    "moh_all",
    "moh_half",
]

missing_models = [
    model
    for model in required_models
    if model not in df["model"].values
]

if missing_models:

    raise ValueError(
        "Missing benchmark results for: "
        + ", ".join(missing_models)
    )


# =====================================================================
# HELPER
# =====================================================================

def get_model_row(
    model_name,
):

    return (
        df[
            df["model"]
            == model_name
        ]
        .iloc[0]
    )


# =====================================================================
# FIGURE
# =====================================================================

fig, ax = plt.subplots(
    figsize=(10, 6.5)
)


# =====================================================================
# BASELINE
# =====================================================================

baseline = get_model_row(
    "baseline"
)

baseline_x = baseline[
    "ms_per_triplet"
]

baseline_y = baseline[
    "test_2afc"
]


# ---------------------------------------------------------------------
# Baseline reference lines
# ---------------------------------------------------------------------

ax.axhline(
    baseline_y,
    linestyle="--",
    linewidth=1,
    color=BASELINE_COLOR,
    alpha=0.45,
    zorder=1,
)

ax.axvline(
    baseline_x,
    linestyle="--",
    linewidth=1,
    color=BASELINE_COLOR,
    alpha=0.45,
    zorder=1,
)


# ---------------------------------------------------------------------
# Baseline point
# ---------------------------------------------------------------------

ax.scatter(
    baseline_x,
    baseline_y,
    marker="*",
    s=260,
    color=MAIN_COLOR,
    edgecolor="white",
    linewidth=1.2,
    zorder=6,
)

ax.annotate(
    "Baseline",
    (
        baseline_x,
        baseline_y,
    ),
    xytext=(8, 8),
    textcoords="offset points",
    fontsize=11,
    fontweight="bold",
    color=MAIN_COLOR,
)


# =====================================================================
# EFFICIENT ATTENTION FAMILIES
# =====================================================================

label_offsets = {
    "SRA": (8, 8),
    "MetaFormer": (8, -18),
    "MoH": (8, 8),
}


for family_name, models in families.items():

    full = get_model_row(
        models["full"]
    )

    half = get_model_row(
        models["half"]
    )

    x_full = full[
        "ms_per_triplet"
    ]

    y_full = full[
        "test_2afc"
    ]

    x_half = half[
        "ms_per_triplet"
    ]

    y_half = half[
        "test_2afc"
    ]

    # -----------------------------------------------------------------
    # Arrow:
    #
    # Full  --->  Half
    # -----------------------------------------------------------------

    ax.annotate(
        "",
        xy=(
            x_half,
            y_half,
        ),
        xytext=(
            x_full,
            y_full,
        ),
        arrowprops=dict(
            arrowstyle="->",
            color=MAIN_COLOR,
            linewidth=2,
            alpha=0.75,
            shrinkA=8,
            shrinkB=8,
        ),
        zorder=2,
    )

    # -----------------------------------------------------------------
    # FULL
    #
    # Empty marker
    # -----------------------------------------------------------------

    ax.scatter(
        x_full,
        y_full,
        marker="o",
        s=120,
        facecolor="white",
        edgecolor=MAIN_COLOR,
        linewidth=2.3,
        zorder=5,
    )

    # -----------------------------------------------------------------
    # HALF
    #
    # Filled marker
    # -----------------------------------------------------------------

    ax.scatter(
        x_half,
        y_half,
        marker="o",
        s=120,
        color=MAIN_COLOR,
        edgecolor="white",
        linewidth=1,
        zorder=5,
    )

    # -----------------------------------------------------------------
    # Family label beside Half point
    # -----------------------------------------------------------------

    offset = label_offsets[
        family_name
    ]

    ax.annotate(
        family_name,
        (
            x_half,
            y_half,
        ),
        xytext=offset,
        textcoords="offset points",
        fontsize=11,
        fontweight="bold",
        color=MAIN_COLOR,
    )


# =====================================================================
# FULL / HALF LEGEND
# =====================================================================

ax.scatter(
    [],
    [],
    marker="*",
    s=180,
    color=MAIN_COLOR,
    label="Baseline",
)

ax.scatter(
    [],
    [],
    marker="o",
    s=90,
    facecolor="white",
    edgecolor=MAIN_COLOR,
    linewidth=2,
    label="Full",
)

ax.scatter(
    [],
    [],
    marker="o",
    s=90,
    color=MAIN_COLOR,
    label="Half",
)


# =====================================================================
# LABELS
# =====================================================================

ax.set_xlabel(
    "Inference latency (ms / triplet)",
    fontsize=12,
)

ax.set_ylabel(
    "2AFC accuracy (%)",
    fontsize=12,
)

ax.set_title(
    "Accuracy–Latency Trade-off",
    fontsize=15,
    fontweight="bold",
)


# =====================================================================
# GRID
# =====================================================================

ax.grid(
    True,
    linestyle=":",
    linewidth=0.8,
    alpha=0.22,
)


# =====================================================================
# AXIS LIMITS
#
# Automatically zoom around the measured data.
# =====================================================================

x_values = df[
    df["model"].isin(
        required_models
    )
]["ms_per_triplet"]

y_values = df[
    df["model"].isin(
        required_models
    )
]["test_2afc"]


x_range = (
    x_values.max()
    - x_values.min()
)

y_range = (
    y_values.max()
    - y_values.min()
)


x_padding = max(
    x_range * 0.15,
    0.05,
)

y_padding = max(
    y_range * 0.12,
    1.0,
)


ax.set_xlim(
    x_values.min()
    - x_padding,
    x_values.max()
    + x_padding,
)

ax.set_ylim(
    y_values.min()
    - y_padding,
    y_values.max()
    + y_padding,
)


# =====================================================================
# LEGEND
# =====================================================================

ax.legend(
    loc="lower right",
    frameon=False,
    fontsize=10,
)


# =====================================================================
# SMALL DIRECTION INDICATOR
# =====================================================================

ax.text(
    0.02,
    0.97,
    "Better ↑  ← Faster",
    transform=ax.transAxes,
    fontsize=10,
    fontweight="bold",
    color=MAIN_COLOR,
    verticalalignment="top",
)


# =====================================================================
# STYLE
# =====================================================================

ax.spines[
    "top"
].set_visible(
    False
)

ax.spines[
    "right"
].set_visible(
    False
)

ax.spines[
    "left"
].set_alpha(
    0.4
)

ax.spines[
    "bottom"
].set_alpha(
    0.4
)

plt.tight_layout()


# =====================================================================
# SAVE
# =====================================================================

plt.savefig(
    OUTPUT_PNG,
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

plt.savefig(
    OUTPUT_PDF,
    bbox_inches="tight",
    facecolor="white",
)

print(
    f"Saved PNG: {OUTPUT_PNG}"
)

print(
    f"Saved PDF: {OUTPUT_PDF}"
)