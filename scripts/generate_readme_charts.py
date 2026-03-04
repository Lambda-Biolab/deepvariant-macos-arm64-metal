#!/usr/bin/env python3
"""Generate README charts for the DeepVariant macOS ARM64 fork.

Produces three PNGs from hardcoded empirical measurements (M1 Max, HG003 chr20):
  docs/images/optimization_waterfall.png  — cumulative optimization journey
  docs/images/platform_comparison.png     — M1 Max vs GCP instances (impact view)
  docs/images/cost_comparison.png         — cumulative cost: Apple Silicon vs GCP cloud

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
    {"label": "+ CoreML\n(GPU + Neural Engine)",      "me": 263, "cv": 175, "pp": 16, "fp": False},
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

# Directly measured: GCP Cloud Run + NVIDIA L4 (24 GB VRAM, 8 vCPU, 32 GB RAM)
# DeepVariant v1.9.0 GPU Docker (deepvariant_gpu:latest), HG003 chr20, 8 shards. Measured March 2026.
# Same pipeline as M1 Max benchmark (--call_small_model_examples enabled in both).
GCP_L4_TOTAL    = 584   # seconds (measured)
GCP_L4_ME       = 484   # make_examples: 8 shards on 8 Intel vCPUs
GCP_L4_CV       = 64    # call_variants: NVIDIA L4 GPU (24 GB VRAM)
GCP_L4_PP       = 20    # postprocess_variants

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
    """Horizontal bar chart: M1 Max vs GCP — includes measured L4 GPU data."""
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
        ("GCP Cloud Run + L4 GPU\n(8 vCPU, 24 GB VRAM) ★ DIRECTLY MEASURED",
         GCP_L4_TOTAL, "#9B59B6", "normal", None),
        ("GCP n2-standard-16 + P100 GPU\n(est. sequential, Google-recommended GPU)",
         GCP_16_GPU_EST, "#F39C12", "normal", "est."),
        ("GCP n2-standard-16\n(16 vCPU, CPU-only) ★ DIRECTLY MEASURED",
         GCP_16_CHR20_MEASURED, "#E74C3C", "normal", None),
    ]

    fig, ax = plt.subplots(figsize=(13, 6.2))
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
    l4_idx  = 3  # L4 GPU row index
    speedup_fp  = GCP_16_CHR20_MEASURED / FAST_PIPELINE_TOTAL   # 3.88×
    speedup_seq = GCP_16_CHR20_MEASURED / m1_sequential          # 2.09×
    speedup_l4  = GCP_L4_TOTAL / FAST_PIPELINE_TOTAL             # M1 vs L4 total
    speedup_gpu = GCP_16_GPU_EST / FAST_PIPELINE_TOTAL           # M1 vs P100 est.

    # Callout box — placed to the right of the M1 Max fast pipeline bar at the same row.
    # xlim is expanded to 1.55× GCP-CPU to give enough room for the box without crowding.
    ax.annotate(
        f"  {speedup_fp:.2f}× faster than GCP 16-vCPU (measured)\n"
        f"  {speedup_l4:.2f}× faster than GCP L4 GPU (measured)\n"
        f"  {speedup_gpu:.2f}× faster than GCP P100 GPU (est.)",
        xy=(FAST_PIPELINE_TOTAL, fp_idx),
        xytext=(GCP_16_CHR20_MEASURED * 0.62, fp_idx),
        fontsize=8.5, fontweight="bold", color="#155724",
        va="center",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#d4edda",
                  edgecolor="#28a745", linewidth=1.2),
        arrowprops=dict(arrowstyle="->", color="#28a745", lw=1.2),
    )

    # L4 GPU note — call_variants comparison
    cv_speedup_l4 = GCP_L4_CV / (STEPS[1]["cv"])  # L4 CV / M1 Metal CV = faster
    ax.text(GCP_L4_TOTAL + 12, l4_idx - 0.38,
            f"CV: {GCP_L4_CV}s ({(STEPS[1]['cv']/GCP_L4_CV):.1f}× faster CV but slower ME)",
            fontsize=7.5, color="#6C3483")

    ax.set_yticks(range(n))
    ax.set_yticklabels([p[0] for p in platforms], fontsize=9.5)
    ax.invert_yaxis()

    ax.set_xlabel("Total pipeline time — HG003 chr20 (seconds)", fontsize=11)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.set_xlim(0, GCP_16_CHR20_MEASURED * 1.55)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    ax.set_title(
        "DeepVariant v1.9 — Apple M1 Max vs Google Cloud Instances\n"
        "HG003 chr20 | Total pipeline wall time",
        fontsize=13, pad=12,
    )

    fig.text(
        0.01, 0.01,
        "★ GCP n2-standard-16 (Intel Xeon @ 2.80 GHz, 16 vCPUs): "
        "directly measured March 2026 — ME 377s, CV 450s, PP 41s.  "
        "★ GCP Cloud Run L4 GPU (NVIDIA L4 24 GB VRAM, 8 vCPU): "
        "directly measured March 2026 — ME 484s, CV 64s, PP 20s.  "
        "P100 est. uses Google's 2.5× speedup.  GCP 96-vCPU from docs/metrics.md scaled to chr20.",
        fontsize=7.5, color="gray", ha="left",
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Chart C: Cost comparison — cumulative cost vs samples processed
# ---------------------------------------------------------------------------

# GCP pricing (on-demand, us-central1, March 2026)
GCP_CPU_RATE_HR   = 0.7769   # n2-standard-16 $/hr
GCP_CPU_TIME_S    = 868      # measured pipeline time (seconds)
GCP_CPU_PER_SAMPLE = GCP_CPU_RATE_HR * (GCP_CPU_TIME_S / 3600)   # ~$0.187

# Cloud Run + L4: 8 vCPU × $0.000080/vCPU/s + 32 GiB × $0.0000090/GiB/s + L4 × $0.000187/GPU/s
GCP_L4_TIME_S    = 584       # measured
GCP_L4_PER_SAMPLE = (8 * 0.000080 + 32 * 0.0000090 + 0.000187) * GCP_L4_TIME_S   # ~$0.651

# Apple Silicon marginal cost: electricity only (~70W × time × $0.15/kWh)
M1_ELEC_PER_SAMPLE = 0.070 * (224 / 3600) * 0.15   # ~$0.00065 ≈ $0 effectively


def plot_cost_comparison(output_path, show=False, dpi=150):
    """Cumulative cost chart: GCP cloud vs Apple Silicon (owned vs purchased)."""
    max_samples = 10_000
    samples = np.linspace(0, max_samples, 500)

    # Hardware options: (label, upfront_cost, color, linestyle, zorder)
    hardware = [
        ("GCP n2-standard-16 (CPU-only)\n$0.19/sample",
         0, GCP_CPU_PER_SAMPLE, "#E74C3C", "-", 4),
        ("GCP Cloud Run + NVIDIA L4 GPU\n$0.65/sample",
         0, GCP_L4_PER_SAMPLE, "#9B59B6", "-", 4),
        ("Apple Silicon (already owned)\n~$0/sample (electricity only)",
         0, M1_ELEC_PER_SAMPLE, "#27AE60", "-", 5),
        ("Used M1 Max (~$1,500)\n$0.001/sample after purchase",
         1500, M1_ELEC_PER_SAMPLE, "#2980B9", "--", 3),
        ("Mac Studio M2 Ultra (~$2,500 used)\n$0.001/sample after purchase",
         2500, M1_ELEC_PER_SAMPLE, "#1ABC9C", "--", 3),
    ]

    fig, ax = plt.subplots(figsize=(12, 6.5))

    lines = []
    for label, upfront, per_sample, color, ls, zo in hardware:
        y = upfront + per_sample * samples
        line, = ax.plot(samples, y, color=color, linestyle=ls, linewidth=2.2,
                        zorder=zo, label=label)
        lines.append(line)

    # Break-even annotations
    be_l4  = 1500 / (GCP_L4_PER_SAMPLE - M1_ELEC_PER_SAMPLE)   # ~2,308
    be_cpu = 1500 / (GCP_CPU_PER_SAMPLE - M1_ELEC_PER_SAMPLE)   # ~8,021
    be_l4_cost = 1500 + M1_ELEC_PER_SAMPLE * be_l4

    ax.axvline(be_l4, color="#2980B9", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.annotate(
        f"Break-even vs L4 GPU\n~{be_l4:.0f} samples",
        xy=(be_l4, be_l4_cost),
        xytext=(be_l4 + 300, be_l4_cost + 300),
        fontsize=8.5, color="#2980B9",
        arrowprops=dict(arrowstyle="->", color="#2980B9", lw=1.0),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#2980B9", alpha=0.9),
    )

    if be_cpu <= max_samples:
        be_cpu_cost = 1500 + M1_ELEC_PER_SAMPLE * be_cpu
        ax.axvline(be_cpu, color="#E74C3C", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.annotate(
            f"Break-even vs CPU\n~{be_cpu:.0f} samples",
            xy=(be_cpu, be_cpu_cost),
            xytext=(be_cpu + 250, be_cpu_cost - 600),
            fontsize=8.5, color="#E74C3C",
            arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=1.0),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#E74C3C", alpha=0.9),
        )

    ax.set_xlabel("Cumulative samples processed (HG003 chr20 equivalent)", fontsize=11)
    ax.set_ylabel("Cumulative cost (USD)", fontsize=11)
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_xlim(0, max_samples)
    ax.set_ylim(0, max_samples * GCP_L4_PER_SAMPLE * 1.05)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)

    ax.legend(fontsize=9, loc="upper left", framealpha=0.93)

    ax.set_title(
        "Cumulative Cost: Apple Silicon vs GCP Cloud Compute\n"
        "DeepVariant v1.9 — HG003 chr20 equivalent workload",
        fontsize=13, pad=12,
    )

    fig.text(
        0.01, 0.01,
        "GCP pricing: on-demand us-central1 (March 2026). n2-standard-16: $0.777/hr, measured 868s/sample. "
        "Cloud Run L4: 8 vCPU ($0.000080/vCPU/s) + 32 GiB ($0.0000090/GiB/s) + L4 ($0.000187/GPU/s), measured 584s/sample. "
        "Apple Silicon marginal cost: electricity only (~70W × 224s × $0.15/kWh ≈ $0.001/sample). "
        "Hardware prices: used market estimates (March 2026). Preemptible/spot VMs reduce GCP costs ~60–80% (with interruption risk).",
        fontsize=7.0, color="gray", ha="left", wrap=True,
    )

    plt.tight_layout(rect=[0, 0.07, 1, 1])
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
    plot_cost_comparison(
        os.path.join(output_dir, "cost_comparison.png"),
        show=args.show, dpi=args.dpi,
    )
    print(f"\nDone — 3 charts saved to {output_dir}/")


if __name__ == "__main__":
    main()
