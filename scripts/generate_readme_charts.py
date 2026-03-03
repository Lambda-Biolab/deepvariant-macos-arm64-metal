#!/usr/bin/env python3
"""Generate README charts for the DeepVariant macOS ARM64 fork.

Produces two PNGs from hardcoded empirical measurements (M1 Max, HG003 chr20):
  docs/images/optimization_waterfall.png  — cumulative optimization journey
  docs/images/platform_comparison.png     — M1 Max vs GCP instances (impact view)

All data is hardcoded; no external JSON file required.

Usage:
    python3 scripts/generate_readme_charts.py           # save to docs/images/
    python3 scripts/generate_readme_charts.py --show    # also display interactively
    python3 scripts/generate_readme_charts.py -o /tmp/  # custom output dir
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Measured data — M1 Max (8 perf cores, 32-core GPU, 32 GB RAM), HG003 chr20
# ---------------------------------------------------------------------------

# Sequential optimization steps.
# me/cv/pp = wall time in seconds for each stage.
# fp=True means this step uses the fast pipeline (ME+CV run concurrently).
STEPS = [
    {"label": "CPU-only\nbaseline",                 "me": 263, "cv": 950, "pp": 16, "fp": False},
    {"label": "+ Metal GPU\n(tensorflow-metal)",     "me": 263, "cv": 224, "pp": 16, "fp": False},
    {"label": "+ CoreML\n(Apple Neural Engine)",     "me": 263, "cv": 175, "pp": 16, "fp": False},
    {"label": "+ Haplotype cap\n(−14.7% ME)",        "me": 224, "cv": 175, "pp": 16, "fp": False},
    {"label": "+ Fast pipeline\n(ME+CV concurrent)", "me": 224, "cv": 175, "pp": 16, "fp": True},
]

# Measured total wall time for the fast pipeline + CoreML + haplotype cap run.
# Filled in from benchmark.sh --use-coreml --fast-pipeline after the haplotype cap commit.
FAST_PIPELINE_TOTAL = 224  # seconds (measured: benchmark.sh --use-coreml --fast-pipeline, post-haplotype-cap)

# CPU-only sequential total (used as speedup denominator)
CPU_ONLY_TOTAL = STEPS[0]["me"] + STEPS[0]["cv"] + STEPS[0]["pp"]  # 1229s

# ---------------------------------------------------------------------------
# GCP reference constants
# ---------------------------------------------------------------------------

CHR20_SCALE = 64_444_167 / 3_088_286_401  # chr20 fraction of whole genome

# Published full-genome timings (n2-standard-96, 96 vCPU, CPU-only)
# from docs/metrics.md
PUBLISHED_FG_96 = {
    "make_examples": 45 * 60 + 14,
    "call_variants": 16 * 60 + 26,
    "postprocess_variants": 6 * 60 + 51,
    "total": 78 * 60 + 58,
}

GCP_96_CHR20 = PUBLISHED_FG_96["total"] * CHR20_SCALE    # ≈ 99s

# Directly measured: GCP n2-standard-16 (Intel Xeon @ 2.80 GHz, 16 vCPUs / 8 physical cores)
# DeepVariant v1.9.0 Docker, HG003 chr20, CPU-only. Measured March 2026.
GCP_16_CHR20_MEASURED = 868   # seconds: 377 (ME) + 450 (CV) + 41 (PP)
GCP_16_ME_MEASURED    = 377
GCP_16_CV_MEASURED    = 450
GCP_16_PP_MEASURED    = 41

# Conservative GPU estimate: GCP n2-standard-16 + 1× NVIDIA P100
# P100 provides ~2.5× call_variants speedup (Google docs). No fast pipeline.
GCP_16_GPU_EST = GCP_16_ME_MEASURED + int(GCP_16_CV_MEASURED / 2.5) + GCP_16_PP_MEASURED  # 377+180+41 = 598s

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

COLOR_ME = "#4472C4"   # blue — make_examples
COLOR_CV = "#ED7D31"   # orange — call_variants
COLOR_PP = "#70AD47"   # green — postprocess_variants
COLOR_FP = "#B4C7E7"   # light blue hatched — fast pipeline (concurrent)


def format_time(seconds):
    seconds = round(seconds)
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


# ---------------------------------------------------------------------------
# Chart A: Optimization waterfall
# ---------------------------------------------------------------------------

def plot_optimization_waterfall(output_path, show=False, dpi=150):
    """Stacked horizontal bars showing cumulative optimization journey."""
    n = len(STEPS)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    bar_height = 0.55

    for i, step in enumerate(STEPS):
        if step["fp"]:
            # Fast pipeline: single hatched bar for the measured wall time
            ax.barh(i, FAST_PIPELINE_TOTAL, bar_height,
                    color=COLOR_FP, hatch="//", edgecolor="#6F9DC8", linewidth=0.8,
                    label="ME + CV (concurrent)" if i == n - 1 else None)
            # Postprocess segment at the far right (always sequential)
            ax.barh(i, step["pp"], bar_height,
                    left=FAST_PIPELINE_TOTAL - step["pp"],
                    color=COLOR_PP, edgecolor="white", linewidth=0.5)
            total = FAST_PIPELINE_TOTAL
        else:
            total = step["me"] + step["cv"] + step["pp"]
            ax.barh(i, step["me"], bar_height, color=COLOR_ME,
                    edgecolor="white", linewidth=0.5)
            ax.barh(i, step["cv"], bar_height, left=step["me"],
                    color=COLOR_CV, edgecolor="white", linewidth=0.5)
            ax.barh(i, step["pp"], bar_height, left=step["me"] + step["cv"],
                    color=COLOR_PP, edgecolor="white", linewidth=0.5)

        # Speedup annotation to the right of each bar
        speedup = CPU_ONLY_TOTAL / total
        label = f"  {format_time(total)}   {speedup:.2f}x faster"
        ax.text(total + 15, i, label, va="center", fontsize=9.5, fontweight="bold",
                color="#1a1a1a")

    # Y-axis labels
    ax.set_yticks(range(n))
    ax.set_yticklabels([s["label"] for s in STEPS], fontsize=10)
    ax.invert_yaxis()  # step 0 at top

    # X-axis
    ax.set_xlabel("Total pipeline time (seconds)", fontsize=11)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.set_xlim(0, CPU_ONLY_TOTAL * 1.30)

    # Grid
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    # Legend
    legend_handles = [
        mpatches.Patch(color=COLOR_ME, label="make_examples"),
        mpatches.Patch(color=COLOR_CV, label="call_variants"),
        mpatches.Patch(color=COLOR_PP, label="postprocess_variants"),
        mpatches.Patch(facecolor=COLOR_FP, hatch="//", edgecolor="#6F9DC8",
                       label="fast pipeline (ME+CV concurrent)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9,
              framealpha=0.9)

    ax.set_title(
        "DeepVariant Performance Journey — Apple M1 Max (HG003 chr20)\n"
        "Sequential wall time at each optimization step",
        fontsize=13, pad=12,
    )

    # Footer note
    fig.text(0.01, 0.01,
             "* Speedup vs CPU-only sequential baseline (20m29s).  "
             "Fast pipeline: make_examples and call_variants run concurrently; "
             "total = wall time, not sum of stages.",
             fontsize=7.5, color="gray", ha="left")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Chart B: Platform comparison — impact view
# ---------------------------------------------------------------------------

def plot_platform_comparison(output_path, show=False, dpi=150):
    """Horizontal bar chart: M1 Max vs GCP — emphasises the 3.88× advantage."""
    m1_sequential = STEPS[3]["me"] + STEPS[3]["cv"] + STEPS[3]["pp"]  # 415s

    # Ordered fastest → slowest (top → bottom when axis is inverted)
    platforms = [
        # (label, seconds, color, style, annotation)
        ("GCP n2-standard-96\n(96 vCPU, 12× cores, sequential)",
         GCP_96_CHR20, "#95A5A6", "normal", "12× more cores"),
        ("Apple M1 Max — fast pipeline\n(CoreML + haplotype cap)",
         FAST_PIPELINE_TOTAL, "#2ECC71", "bold", None),
        ("Apple M1 Max — sequential\n(CoreML + haplotype cap)",
         m1_sequential, "#3498DB", "normal", None),
        ("GCP n2-standard-16 + P100 GPU\n(est. sequential, Google-recommended GPU)",
         GCP_16_GPU_EST, "#F39C12", "normal", "est."),
        ("GCP n2-standard-16\n(16 vCPU, CPU-only) ★ DIRECTLY MEASURED",
         GCP_16_CHR20_MEASURED, "#E74C3C", "normal", None),
    ]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    bar_height = 0.52
    n = len(platforms)

    for i, (label, t, color, weight, note) in enumerate(platforms):
        ax.barh(i, t, bar_height, color=color,
                edgecolor="white", linewidth=0.5, zorder=3)
        # Time label
        suffix = f"  ({note})" if note else ""
        ax.text(t + 12, i, f"  {format_time(t)}{suffix}",
                va="center", fontsize=9.5,
                fontweight=weight, color="#1a1a1a")

    # Dashed reference line at GCP 16-vCPU measured
    ax.axvline(GCP_16_CHR20_MEASURED, color="#E74C3C", linestyle="--", linewidth=1.0,
               zorder=4, alpha=0.5)

    # Speedup callouts for the two M1 Max rows
    fp_idx = 1  # fast pipeline row index
    seq_idx = 2  # sequential row index
    speedup_fp  = GCP_16_CHR20_MEASURED / FAST_PIPELINE_TOTAL   # 3.88×
    speedup_seq = GCP_16_CHR20_MEASURED / m1_sequential          # 2.09×
    speedup_gpu = GCP_16_GPU_EST / FAST_PIPELINE_TOTAL           # 2.67×

    # Callout box for fast pipeline
    ax.annotate(
        f"  {speedup_fp:.2f}× faster than GCP 16-vCPU (measured)\n"
        f"  {speedup_gpu:.2f}× faster than GCP 16-vCPU + GPU (est.)",
        xy=(FAST_PIPELINE_TOTAL, fp_idx),
        xytext=(GCP_16_CHR20_MEASURED * 0.38, fp_idx - 0.45),
        fontsize=9, fontweight="bold", color="#155724",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#d4edda",
                  edgecolor="#28a745", linewidth=1.2),
        arrowprops=dict(arrowstyle="->", color="#28a745", lw=1.2),
    )

    # Sequential speedup
    ax.text(m1_sequential + 12, seq_idx - 0.38,
            f"{speedup_seq:.2f}× faster",
            fontsize=8.5, color="#1a5276", fontweight="bold")

    ax.set_yticks(range(n))
    ax.set_yticklabels([p[0] for p in platforms], fontsize=9.5)
    ax.invert_yaxis()

    ax.set_xlabel("Total pipeline time — HG003 chr20 (seconds)", fontsize=11)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.set_xlim(0, GCP_16_CHR20_MEASURED * 1.30)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    ax.set_title(
        "DeepVariant v1.9 — Apple M1 Max vs Google Cloud Instances\n"
        "HG003 chr20 | Total pipeline wall time",
        fontsize=13, pad=12,
    )

    fig.text(
        0.01, 0.01,
        "★ GCP n2-standard-16 (Intel Xeon @ 2.80 GHz, 16 vCPUs / 8 phys. cores): "
        "directly measured March 2026 — make_examples 377s, call_variants 450s, postprocess 41s.  "
        "GPU est. uses Google's published 2.5× P100 speedup on call_variants (no fast pipeline).  "
        "GCP 96-vCPU from docs/metrics.md scaled to chr20 (64M / 3.1G bases).",
        fontsize=7.5, color="gray", ha="left",
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate README charts for the DeepVariant macOS ARM64 fork"
    )
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Output directory (default: docs/images/ relative to repo root)")
    parser.add_argument("--show", action="store_true",
                        help="Display charts interactively")
    parser.add_argument("--dpi", type=int, default=150,
                        help="DPI for saved images (default: 150)")
    args = parser.parse_args()

    # Find repo root (directory containing README.md)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.exists(os.path.join(repo_root, "README.md")):
        # Fallback: current working directory
        repo_root = os.getcwd()

    output_dir = args.output_dir or os.path.join(repo_root, "docs", "images")
    os.makedirs(output_dir, exist_ok=True)

    plot_optimization_waterfall(
        os.path.join(output_dir, "optimization_waterfall.png"),
        show=args.show, dpi=args.dpi,
    )
    plot_platform_comparison(
        os.path.join(output_dir, "platform_comparison.png"),
        show=args.show, dpi=args.dpi,
    )
    print(f"\nDone — 2 charts saved to {output_dir}/")


if __name__ == "__main__":
    main()
