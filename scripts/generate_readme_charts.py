#!/usr/bin/env python3
"""Generate README charts for the DeepVariant macOS ARM64 fork.

Produces two PNGs from hardcoded empirical measurements (M1 Max, HG003 chr20):
  docs/images/optimization_waterfall.png  — cumulative optimization journey
  docs/images/platform_comparison.png     — M1 Max vs GCP instances

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
# GCP reference constants — identical to benchmark_viz.py
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

# Estimated full-genome timings for n2-standard-16 (8 physical cores)
# from DeepVariant-on-Spark paper scaling ratios (PMC7481958)
ESTIMATED_FG_16 = {
    "make_examples": int(PUBLISHED_FG_96["make_examples"] * 5.108),
    "call_variants": int(PUBLISHED_FG_96["call_variants"] * 2.820),
    "postprocess_variants": int(PUBLISHED_FG_96["postprocess_variants"] * 1.167),
}
ESTIMATED_FG_16["total"] = sum(ESTIMATED_FG_16[s]
                               for s in ["make_examples", "call_variants", "postprocess_variants"])

GCP_96_CHR20 = PUBLISHED_FG_96["total"] * CHR20_SCALE    # ≈ 99s
GCP_16_CHR20 = ESTIMATED_FG_16["total"] * CHR20_SCALE    # ≈ 358s

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
# Chart B: Platform comparison
# ---------------------------------------------------------------------------

def plot_platform_comparison(output_path, show=False, dpi=150):
    """Horizontal bar chart: M1 Max best vs GCP instances."""
    m1_sequential = STEPS[3]["me"] + STEPS[3]["cv"] + STEPS[3]["pp"]  # 415s

    platforms = [
        # (label, total_seconds, color, is_m1)
        ("GCP n2-standard-96\n(96 vCPU, CPU-only)",
         GCP_96_CHR20, "#95A5A6", False),
        ("GCP n2-standard-16\n(8 phys. cores, CPU-only, est.)",
         GCP_16_CHR20, "#E67E22", False),
        ("M1 Max — sequential\n(CoreML + haplotype cap)",
         m1_sequential, "#3498DB", True),
        ("M1 Max — fast pipeline\n(CoreML + haplotype cap, measured)",
         FAST_PIPELINE_TOTAL, "#2ECC71", True),
    ]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    bar_height = 0.52
    n = len(platforms)

    for i, (label, t, color, is_m1) in enumerate(platforms):
        ax.barh(i, t, bar_height, color=color,
                edgecolor="white", linewidth=0.5,
                zorder=3)
        # Time label at end of bar
        ax.text(t + 8, i, f"  {format_time(t)}",
                va="center", fontsize=10,
                fontweight="bold" if is_m1 else "normal",
                color="#1a1a1a")

    # Vertical dashed line at GCP 16-vCPU time
    ax.axvline(GCP_16_CHR20, color="#E67E22", linestyle="--", linewidth=1.2,
               zorder=4, label="Equivalent-core GCP baseline")

    # Background shading
    ax.axvspan(0, GCP_16_CHR20, alpha=0.04, color="#2ECC71", zorder=1)
    ax.axvspan(GCP_16_CHR20, ax.get_xlim()[1] if ax.get_xlim()[1] > GCP_16_CHR20 else 1200,
               alpha=0.04, color="#E74C3C", zorder=1)

    # "FASTER" / "SLOWER" annotation
    ax.text(GCP_16_CHR20 * 0.5, n - 0.1, "← faster", ha="center",
            fontsize=9, color="#27AE60", fontstyle="italic")
    ax.text(GCP_16_CHR20 * 1.3, n - 0.1, "slower →", ha="center",
            fontsize=9, color="#C0392B", fontstyle="italic")

    ax.set_yticks(range(n))
    ax.set_yticklabels([p[0] for p in platforms], fontsize=10)
    ax.invert_yaxis()

    ax.set_xlabel("Total pipeline time — HG003 chr20 (seconds)", fontsize=11)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.set_xlim(0, max(t for _, t, _, _ in platforms) * 1.25)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9)

    ax.set_title(
        "DeepVariant v1.9 — Apple M1 Max vs GCP Instances\n"
        "HG003 chr20 | Total pipeline wall time",
        fontsize=13, pad=12,
    )

    fig.text(0.01, 0.01,
             "GCP 16-vCPU: estimated from published scaling data (PMC7481958). "
             "GCP 96-vCPU: from docs/metrics.md scaled to chr20 (64M / 3.1G bases). "
             "M1 Max times measured; GCP times are sequential (no fast pipeline).",
             fontsize=7.5, color="gray", ha="left")

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
