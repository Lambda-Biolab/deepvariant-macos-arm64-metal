# DeepVariant — macOS ARM64 (Apple Silicon) Native Build

[![release](https://img.shields.io/badge/base-v1.9.0-green?logo=github)](https://github.com/google/deepvariant/releases)
[![platform](https://img.shields.io/badge/platform-macOS%20ARM64-blue?logo=apple)](https://support.apple.com/en-us/116943)
[![gpu](https://img.shields.io/badge/Metal%20GPU-4.25x%20speedup-orange?logo=apple)](https://developer.apple.com/metal/)
[![coreml](https://img.shields.io/badge/CoreML-1.28x%20on%20top%20of%20GPU-blueviolet?logo=apple)](https://developer.apple.com/documentation/coreml)
[![pipeline](https://img.shields.io/badge/fast%20pipeline%20%2B%20CoreML-4m25s%20total%20%E2%80%94%201.93x%20vs%20sequential-brightgreen?logo=apple)](https://github.com/google/deepvariant/blob/r1.9/docs/deepvariant-fast-pipeline-case-study.md)

This is a fork of [Google DeepVariant](https://github.com/google/deepvariant) v1.9.0 that builds and runs **natively on macOS with Apple Silicon** (M1, M2, M3, M4) — no Docker, no Rosetta, no remote server.

> **Why this matters:** The official DeepVariant Docker image [crashes on Apple Silicon](https://github.com/google/deepvariant/issues/657) because TensorFlow's x86_64 binaries require AVX instructions that Rosetta 2 cannot translate inside Docker's Linux VM. Rebuilding the image for `linux/arm64` is theoretically possible but yields a CPU-only container with no GPU acceleration — more than 4x slower. There is no official macOS build. Before this fork, the only practical option was a remote Linux server. This fork patches the Bazel build system, C++ source, and third-party dependencies to compile and run natively on macOS ARM64 with Metal GPU acceleration.

## What is DeepVariant?

DeepVariant is a deep learning-based variant caller that takes aligned reads (in BAM or CRAM format), produces pileup image tensors from them, classifies each tensor using a convolutional neural network, and finally reports the results in a standard VCF or gVCF file.

DeepVariant supports germline variant-calling in diploid organisms. For full documentation on DeepVariant's capabilities, case studies, and supported data types, see the [upstream repository](https://github.com/google/deepvariant).

---

## Quick Install (Pre-built Binaries)

Install pre-built binaries with a single command. No build tools required.

### Homebrew (Recommended)

**Prerequisites:** **macOS** on Apple Silicon (M1/M2/M3/M4) with [Homebrew](https://brew.sh) installed.

```bash
brew tap antomicblitz/deepvariant
brew install deepvariant
```

Download a model and verify:

```bash
deepvariant-download-model WGS    # ~200 MB download + CoreML conversion (~4 min total)
deepvariant-quicktest              # end-to-end verification
```

`deepvariant-download-model` automatically converts the WGS model to CoreML format on Apple Silicon — no extra step needed. `run_deepvariant` then uses both CoreML and fast pipeline automatically, giving a **combined ~5.43x speedup** over CPU baseline: Metal GPU (4.25x), CoreML on top (×1.28), and fast pipeline running `make_examples` and `call_variants` concurrently (×1.93 total wall time vs sequential). On M1 Max this processes HG003 chr20 in **4m25s** with zero accuracy loss. To skip CoreML conversion: `SKIP_COREML=1 deepvariant-download-model WGS`.

Run DeepVariant:

```bash
run_deepvariant \
  --model_type=WGS \
  --ref=reference.fasta \
  --reads=input.bam \
  --output_vcf=output.vcf \
  --num_shards=$(sysctl -n hw.ncpu)
```

Download additional models any time:

```bash
deepvariant-download-model WES PACBIO ONT_R104
deepvariant-download-model WGS --deeptrio
```

Available models: WGS, WES, PACBIO, ONT_R104, HYBRID, MASSEQ

Uninstall:

```bash
brew uninstall deepvariant
brew untap antomicblitz/deepvariant
```

### conda / venv Install

#### conda

**Prerequisites:** **conda**, **mamba**, or **micromamba** — native ARM64 (e.g. [Miniforge](https://github.com/conda-forge/miniforge))

```bash
curl -fsSL https://raw.githubusercontent.com/antomicblitz/deepvariant-macos-arm64-metal/r1.9/install.sh | USE_CONDA=1 bash
```

This creates a `deepvariant` conda environment with Python 3.10, GNU parallel, and all dependencies (including `coremltools`). After downloading the WGS model, the installer automatically converts it to CoreML format (~2 min, one-time) for the **1.28x additional speedup**. To skip: `SKIP_COREML=1 curl ... | USE_CONDA=1 bash`. Activate the environment with `conda activate deepvariant`.

Alternatively, create the environment manually from the included `environment.yml`:

```bash
git clone https://github.com/antomicblitz/deepvariant-macos-arm64-metal.git
cd deepvariant-macos-arm64-metal
conda env create -f environment.yml
conda activate deepvariant
# Then install --no-deps packages:
pip install --no-deps tensorflow-hub==0.14.0 tensorflow-model-optimization==0.7.5 tf-models-official==2.13.1
```

#### venv

If you already have **Python 3.10** installed (e.g., `brew install python@3.10`), the installer can use a lightweight venv instead of conda:

```bash
curl -fsSL https://raw.githubusercontent.com/antomicblitz/deepvariant-macos-arm64-metal/r1.9/install.sh | bash
```

> **Note:** Python 3.10 specifically is required — `tensorflow-macos 2.13.1` does not support other Python versions. You also need GNU parallel installed separately (`brew install parallel`).

Like the conda path, the venv installer includes `coremltools` and auto-converts the WGS model to CoreML after download. Both install paths share the same package installation and CoreML conversion steps.

#### Environment Variables

Customize the `install.sh` script with environment variables:

```bash
# Install to a custom location
curl -fsSL ... | DEEPVARIANT_HOME=/path/to/dir bash

# Download specific models (WGS WES PACBIO ONT_R104 HYBRID MASSEQ ALL NONE)
curl -fsSL ... | MODEL_TYPES="WGS WES PACBIO" bash

# Force conda or venv
curl -fsSL ... | USE_CONDA=1 bash
curl -fsSL ... | USE_CONDA=0 bash   # force venv, fail if no Python 3.10

# Custom conda env name (default: deepvariant)
curl -fsSL ... | CONDA_ENV_NAME=dv19 USE_CONDA=1 bash

# Skip environment creation entirely (if you manage your own)
curl -fsSL ... | SKIP_ENV=1 bash

# Skip CoreML model conversion (e.g. for non-WGS workflows)
curl -fsSL ... | SKIP_COREML=1 bash
```

#### After Installation

Open a new terminal, activate your environment (`conda activate deepvariant` or `source ~/.deepvariant/venv/bin/activate`), and run:

```bash
run_deepvariant \
  --model_type=WGS \
  --ref=reference.fasta \
  --reads=input.bam \
  --output_vcf=output.vcf \
  --num_shards=$(sysctl -n hw.ncpu)
```

Download additional models any time:

```bash
deepvariant-download-model WES PACBIO ONT_R104
deepvariant-download-model WGS --deeptrio
```

#### Quicktest

Verify your installation with a small end-to-end test (requires GNU parallel):

```bash
$DEEPVARIANT_HOME/scripts/quicktest.sh
```

This runs all three DeepVariant steps on a 10kb region of chr20, confirms Metal GPU detection, and produces a VCF output.

#### Uninstalling

```bash
deepvariant-uninstall
```

This removes the install directory, conda/venv environment, shell profile entries, and quicktest data. It shows what will be removed and asks for confirmation before proceeding.

---

## Build from Source

If you prefer to build from source instead of using pre-built binaries:

### Prerequisites

- **macOS** on Apple Silicon (M1/M2/M3/M4)
- **Homebrew** — [https://brew.sh](https://brew.sh)
- **Python 3.10** — `brew install python@3.10`
- **Xcode Command Line Tools** — `xcode-select --install`
- ~30 GB disk space (TensorFlow source + Bazel cache)

### 1. Clone the Repository

```bash
git clone https://github.com/antomicblitz/deepvariant-macos-arm64-metal.git
cd deepvariant-macos-arm64-metal
git checkout r1.9
```

### 2. Install Build Prerequisites

```bash
./build-prereq-macos.sh
```

This installs Homebrew packages, Bazel 5.3.0, abseil-cpp, CLIF C++ runtime, clones/configures TensorFlow 2.13.1 source, and runs `run-prereq-macos.sh` for Python packages.

### 3. Patch zlib in Bazel Cache (Required After `bazel clean`)

After the first Bazel build starts downloading external deps, you must patch zlib's `zutil.h`. This is needed because modern macOS defines `TARGET_OS_MAC`, which causes zlib to set `fdopen` to `NULL`:

```bash
ZUTIL=$(find $(bazel info output_base)/external/zlib -name zutil.h 2>/dev/null)
gsed -i 's/#if defined(MACOS) || defined(TARGET_OS_MAC)/#if defined(MACOS) \&\& !defined(__APPLE__)/' "$ZUTIL"
```

> **Important:** This patch lives in the Bazel cache and must be reapplied after `bazel clean` or any cache wipe.

### 4. Build

```bash
source settings.sh
./build_release_binaries.sh
```

### 5. Set Up a Python Virtual Environment

```bash
python3.10 -m venv ~/dv-venv
source ~/dv-venv/bin/activate
pip install tensorflow-macos==2.13.1 tensorflow-metal==1.0.0
pip install --no-deps tensorflow-hub==0.14.0 tensorflow-model-optimization==0.7.5 tf-models-official==2.13.1
pip install absl-py protobuf==4.21.9 pysam==0.20.0 contextlib2 etils typing_extensions \
  importlib_resources sortedcontainers==2.1.0 intervaltree==3.1.0 ml_collections \
  clu==0.0.9 joblib psutil pandas==1.3.4 Pillow==9.5.0 scikit-learn==1.0.2 \
  jax==0.4.35 opencv-python-headless markupsafe==2.0.1 coremltools
```

### 6. Download Model and Enable CoreML

```bash
# Download the WGS model — CoreML conversion runs automatically on Apple Silicon (~2 min)
bash scripts/deepvariant-download-model WGS

# CoreML conversion is handled automatically by deepvariant-download-model.
# To convert manually (e.g. after moving model files):
python3 scripts/convert_model_coreml.py --model_dir ~/.deepvariant/models/wgs
```

`run_deepvariant` auto-detects the `.mlmodel` and enables CoreML + fast pipeline — no flags needed. To use `call_variants` directly: add `--use_coreml --batch_size 128`.

### 7. Package for Distribution (Optional)

```bash
./scripts/package_release.sh           # create tarball only
./scripts/package_release.sh --release # create tarball + GitHub Release
```

---

## Benchmarks

We benchmarked DeepVariant v1.9.0 on an **Apple M1 Max** (8 performance cores, 32-core GPU, 32 GB RAM) using the standard HG003 chr20 WGS case study and compared against published GCP metrics.

### call_variants: Three Levels of Acceleration

| Mode | call_variants | Speedup |
|------|--------------|---------|
| CPU-only | 15m50s (950s) | baseline |
| **Metal GPU** (tensorflow-metal) | 3m44s (224s) | **4.25x** |
| **Metal GPU + CoreML** | **2m55s (175s)** | **5.43x** |

Metal GPU is enabled by default with `tensorflow-metal`. CoreML adds a further **1.28x** on top by using Apple's Neural Engine/GPU via the native CoreML framework instead of TensorFlow Metal. Both are zero-configuration after installation — `deepvariant-download-model` handles CoreML conversion automatically on Apple Silicon.

### Full Pipeline: M1 Max (HG003 chr20)

| Mode | `make_examples` | `call_variants` | `postprocess_variants` | **Total** | vs Metal GPU |
|------|-----------------|-----------------|------------------------|-----------|--------------|
| CPU-only | ~272s | ~950s | ~16s | **~21m** | — |
| Metal GPU | 272s | 224s | 16s | **8m32s** | baseline |
| Metal GPU + CoreML | 272s | 175s | 16s | **7m43s** | 1.11x |
| **Fast pipeline + CoreML** | *concurrent* | *concurrent* | **16s** | **4m25s** | **1.93x** |

**Fast pipeline** runs `make_examples` and `call_variants` simultaneously using shared memory IPC. With CoreML, `call_variants` keeps pace with `make_examples` in real time — both finish at ~4m10s, with postprocess taking a further 16s. This is the recommended mode for best performance.

Sequential CoreML saves ~49 seconds vs Metal GPU on chr20 (9.6% improvement). Fast pipeline + CoreML cuts total time nearly in half — a **1.93x speedup** — by eliminating the sequential wait between stages. For whole-genome runs, the overlap benefit scales proportionally.

### Performance: M1 Max vs GCP Instances

| | M1 Max (sequential) | M1 Max (fast pipeline) | GCP 16-vCPU (est.) | GCP 96-vCPU |
|--|---------------------|------------------------|---------------------|-------------|
| `make_examples` | 4m32s | *concurrent* | 4m49s | 57s |
| `call_variants` | **2m55s** | *concurrent* | 58s | 21s |
| `postprocess_variants` | 16s | 16s | 10s | 9s |
| **Total** | **~7m43s** | **4m25s** | **~5m57s** | **~1m39s** |
| **vs GCP 16-vCPU** | 0.77x | **1.34x faster** | baseline | — |

*GCP 16-vCPU times are estimated from [published scaling data](https://pmc.ncbi.nlm.nih.gov/articles/PMC7481958/) (16/32/64/96 CPU counts), adjusted for v1.9 improvements. GCP 96-vCPU times are from [docs/metrics.md](docs/metrics.md), scaled from full genome to chr20 (64M / 3.1G bases). n2-standard-16 has 8 physical Intel Cascade Lake cores with hyperthreading (16 vCPUs), matching the M1 Max's 8 physical performance cores. The GCP times assume sequential execution; fast pipeline is available on any platform.*

### Key Findings

- **`make_examples` (CPU-bound, embarrassingly parallel):** M1 Max matches or slightly beats an equivalent-core GCP instance. Apple Silicon's high per-core performance compensates for the lower core count.

- **`call_variants` (TensorFlow/CoreML inference):** Metal GPU provides a **4.25x speedup** over CPU-only (224s vs 950s). CoreML adds a further **1.28x** by running inference through Apple's native framework (175s vs 224s). Without Metal GPU, this stage alone takes ~16 minutes. The M1 Max is slower than GCP 16-vCPU for this stage in isolation, but the stage is significantly shorter with CoreML.

- **Fast pipeline + CoreML:** Running `make_examples` and `call_variants` concurrently via shared memory IPC eliminates the sequential wait between stages. With CoreML, `call_variants` keeps pace with `make_examples` in real time, reducing total pipeline time from 7m43s to **4m25s** — faster than a comparable 16-vCPU cloud instance running sequentially.

- **`postprocess_variants`:** Mostly single-threaded; comparable across platforms.

- **Overall:** The M1 Max with fast pipeline + CoreML processes HG003 chr20 in **4m25s** — beating the estimated GCP n2-standard-16 total time by 1.34x. It cannot match the 96-core GCP instance (~5x faster overall), as expected given the 12:1 core ratio.

### Accuracy Validation

We validated variant call accuracy against the [Genome in a Bottle](https://www.nist.gov/programs-projects/genome-bottle) (GIAB) HG003 truth set (NIST v4.2.1) using [rtg-tools vcfeval](https://github.com/RealTimeGenomics/rtg-tools). Both the Metal GPU and CoreML builds produce calls that match the published x86_64 reference accuracy:

| | SNP | | | INDEL | | |
|---|---|---|---|---|---|---|
| | **Recall** | **Precision** | **F1** | **Recall** | **Precision** | **F1** |
| **macOS ARM64 (Metal GPU)** | 0.9963 | 0.9993 | **0.9978** | 0.9948 | 0.9983 | **0.9966** |
| **macOS ARM64 (Metal GPU + CoreML)** | 0.9963 | 0.9993 | **0.9978** | 0.9948 | 0.9983 | **0.9966** |
| Published reference (GCP x86_64) | 0.9997 | 0.9993 | 0.9995 | 0.9934 | 0.9956 | 0.9945 |

*Region: chr20. Sample: HG003 (NA24149). Truth set: NIST/GIAB v4.2.1 high-confidence calls. Comparison engine: rtg vcfeval with `--output-mode split`. PASS variants only.*

**Key findings:**
- CoreML produces **bit-for-bit identical variant calls** to Metal GPU. The raw CoreML float16 outputs differ by ≤0.05% from TF Metal due to float16 rounding, which is resolved by probability normalization before any variant calling decision is made.
- All F1 scores are within 0.5% of the published reference — no meaningful accuracy loss from the ARM64/Metal GPU/CoreML platform.
- INDEL F1 is slightly *higher* than the published reference (0.9966 vs 0.9945).
- 69,904 true-positive SNPs with only 52 false positives; 10,573 true-positive INDELs with only 18 false positives.

Run the accuracy benchmark yourself:

```bash
# Install rtg-tools (one-time, native ARM64 via Java)
brew tap brewsci/bio && brew install rtg-tools

# Run full benchmark with accuracy evaluation (~10 min + ~5 GB download on first run)
bash scripts/benchmark.sh                                    # TF Metal (sequential)
bash scripts/benchmark.sh --use-coreml                       # CoreML (sequential)
bash scripts/benchmark.sh --use-coreml --fast-pipeline       # CoreML + fast pipeline (recommended)

# Skip accuracy evaluation (performance only)
bash scripts/benchmark.sh --skip-accuracy --use-coreml --fast-pipeline
```

### Why Run DeepVariant on Apple Silicon?

The official DeepVariant Docker image [crashes on Apple Silicon](https://github.com/google/deepvariant/issues/657) because TensorFlow's x86_64 binaries require AVX instructions, which Rosetta 2 cannot translate inside Docker's Linux VM. Rebuilding the Docker image for `linux/arm64` is theoretically possible but yields a CPU-only container with no GPU acceleration — more than 4x slower than this native build. There is no official macOS build. Before this fork, the only practical option was a remote Linux server.

With this native build, Apple Silicon Macs become viable for:

1. **Local development and testing.** Run the full pipeline on your laptop without Docker, cloud instances, or network access. Valuable for pipeline development, parameter tuning, and education.

2. **Small datasets and targeted regions.** Exome, panel, or single-chromosome analyses complete in minutes — fast enough for interactive workflows.

3. **Privacy and data sovereignty.** Clinical or restricted datasets that cannot leave your facility can be processed locally.

4. **Cost.** No cloud compute charges. The Mac you already own can run DeepVariant.

5. **Reproducibility.** A self-contained local environment with no Docker or cloud dependencies.

**Not recommended for:**

- Full whole-genome sequencing at scale (30x WGS). A 96-core cloud instance at ~79 minutes is more practical than the estimated 6-12 hours on a Mac.
- High-throughput batched processing. Use cloud instances or HPC clusters.

### Metal GPU, CoreML, and Fast Pipeline Acceleration

TensorFlow Metal GPU (`tensorflow-metal`) provides a **4.25x speedup** for `call_variants` inference on Apple Silicon. It is a critical component of this build — without it, the inference stage takes ~4x longer.

**CoreML** is Apple's native on-device machine learning inference framework, built into macOS 11+ and optimized for Apple Silicon at the hardware level. Unlike TensorFlow Metal — which uses Metal as a general-purpose GPU compute path — CoreML routes inference directly through the Neural Engine and GPU via Apple's proprietary runtime with significantly lower framework overhead. For workloads like DeepVariant's `call_variants` (repeated batch inference on fixed-shape tensors), CoreML avoids the dispatch and kernel-launch overhead that TensorFlow Metal incurs, providing an additional **1.28x speedup** on top of Metal GPU.

**Fast pipeline** (`fast_pipeline` binary) runs `make_examples` and `call_variants` concurrently using POSIX shared memory IPC. Instead of writing pileup examples to disk as TFRecords, `make_examples` streams them directly into a shared memory buffer that `call_variants` reads in real time. With CoreML, `call_variants` processes batches fast enough to keep pace with `make_examples`, so both stages finish simultaneously — reducing total wall time from the sum of stages to roughly the maximum. On M1 Max (HG003 chr20), this gives **4m25s total** (vs 7m43s sequential CoreML, vs 8m32s sequential Metal GPU).

The CoreML conversion is a one-time step that happens automatically when you run `deepvariant-download-model WGS` on Apple Silicon. The TF SavedModel is exported to a `.mlmodel` file using the `neuralnetwork` backend (not `mlprogram`, which is incompatible with this model in coremltools 7.x). To convert manually (e.g. after copying a model from another machine):

```bash
deepvariant-convert-coreml                               # Homebrew / install.sh
python3 scripts/convert_model_coreml.py                 # source build
```

`run_deepvariant` **auto-enables fast pipeline + CoreML** on Apple Silicon when the `fast_pipeline` binary is present and the CoreML model has been converted. No extra flags are needed — the acceleration is transparent. You can override with `--fast_pipeline=false` to force sequential execution, or `--fast_pipeline=true` to require fast pipeline (fails with a warning if unavailable).

```bash
# Fast pipeline + CoreML (automatic on Apple Silicon — run_deepvariant auto-detects)
run_deepvariant \
  --model_type WGS \
  --ref GRCh38_no_alt_analysis_set.fasta \
  --reads HG003.cram \
  --output_vcf HG003.vcf.gz \
  --num_shards "$(sysctl -n hw.perflevel0.logicalcpu)"

# Force sequential execution (disable fast pipeline)
run_deepvariant ... --fast_pipeline=false

# Direct call_variants with CoreML (advanced / manual pipeline)
~/.deepvariant/bin/call_variants \
  --outfile output.tfrecord.gz \
  --examples examples.tfrecord.gz \
  --checkpoint ~/.deepvariant/models/wgs \
  --use_coreml --batch_size 128
```

### Running the Benchmark Yourself

The benchmark script automatically prevents macOS from sleeping during the run using `caffeinate`. If you run DeepVariant manually on large datasets, consider wrapping your command with `caffeinate -i` to prevent idle sleep from interrupting long-running stages.

```bash
# Full benchmark with accuracy evaluation (downloads ~5 GB of data on first run)
bash scripts/benchmark.sh                                     # Metal GPU, sequential
bash scripts/benchmark.sh --use-coreml --fast-pipeline        # CoreML + fast pipeline (recommended)

# Performance only, skip accuracy evaluation (~5 min)
bash scripts/benchmark.sh --skip-accuracy --use-coreml --fast-pipeline

# Visualize results
python3 scripts/benchmark_viz.py ~/deepvariant-benchmark/benchmark_results.json --show

# For manual long-running jobs, prevent sleep:
caffeinate -i bash scripts/benchmark.sh --skip-accuracy --use-coreml --fast-pipeline
```

---

## TensorFlow Package Setup

### Correct Package Combination

| Package | Version | Notes |
|---------|---------|-------|
| `tensorflow-macos` | 2.13.1 | Apple's TF build for macOS ARM64 |
| `tensorflow-metal` | 1.0.0 | Metal GPU plugin |

**Do NOT install the standard `tensorflow` pip package alongside `tensorflow-metal`.** Both register the Metal platform, causing a fatal "platform already registered" crash. Use `tensorflow-macos` instead.

### Dependencies that Pull in `tensorflow`

Some packages (e.g., `tensorflow-hub`, `tensorflow-model-optimization`, `tf-models-official`) list `tensorflow` as a dependency and will install the regular package, breaking your setup. Install these with `--no-deps`:

```bash
pip install --no-deps tensorflow-hub==0.14.0
pip install --no-deps tensorflow-model-optimization==0.7.5
pip install --no-deps tf-models-official==2.13.1
```

---

## Architecture

DeepVariant has two distinct layers that matter for this port:

1. **C++ extensions** (`.so` modules) — pileup image generation, allele counting, BAM/VCF I/O, realignment. These are compiled via Bazel against TensorFlow C++ headers at build time.

2. **Python runtime** — ML inference/training via TensorFlow Python API. `tensorflow-metal` registers the Metal GPU device, providing a 4.25x speedup for `call_variants` inference.

The C++ layer only needs TF headers at build time. The Python layer uses pip-installed TensorFlow at runtime.

---

## What Was Changed

This fork modifies the following files from upstream DeepVariant v1.9.0. For the full list, see [the build fixes log](docs/macos-arm64-build-fixes.md).

### New Files

| File | Purpose |
|------|---------|
| `build-prereq-macos.sh` | Homebrew deps, Bazel 5.3.0, abseil-cpp, CLIF runtime, TF source |
| `run-prereq-macos.sh` | Python packages, `tensorflow-macos` + `tensorflow-metal` |
| `third_party/boost.BUILD` | Boost headers from Homebrew (`/opt/homebrew/include`) |
| `scripts/convert_model_coreml.py` | Convert TF SavedModel to CoreML `.mlmodel` for `--use_coreml` |
| `scripts/benchmark.sh` | Full pipeline benchmark with accuracy validation and CoreML support |
| `scripts/benchmark_batch_sizes.sh` | Sweep `call_variants` batch sizes for Metal GPU optimization |
| `scripts/benchmark_hts_threads.sh` | Sweep `--hts_num_threads` for BAM decompression optimization |

### Modified Files

| File | Change |
|------|--------|
| `.bazelrc` | `--config=macos_arm64`, `BOOST_PROCESS_VERSION=1`, `-Wno-unknown-warning-option` |
| `BUILD` | `py_runtime` interpreter path |
| `WORKSPACE` | `new_local_repository` entries for Boost and CLIF at `/opt/homebrew` |
| `settings.sh` | Darwin/ARM64 detection, Homebrew Python paths, macOS-compatible copt flags |
| `build_release_binaries.sh` | BSD `ln`/`sed` compat, `clang++` linking, zsh shebang fix, `PYTHON_BINARY` patch |
| `third_party/htslib.BUILD` | macOS `config.h` (no `fdatasync`, no SSE, `.dylib` plugin extension) |
| `third_party/libssw.BUILD` | `sse2neon.h` for ARM64 SSW (Smith-Waterman) |
| `third_party/sdsl_lite.BUILD` | Removed `"."` from includes (version file conflict) |
| `third_party/clif.BUILD` | Added abseil deps for CLIF C++ runtime |
| `third_party/gbwt.BUILD` | Added `@boost//:boost` dependency |
| `third_party/gbwtgraph.BUILD` | Added `@boost//:boost` dependency |
| `deepvariant/BUILD` | Added `@boost//:boost` to `fast_pipeline`, `stream_examples`, etc. |
| `deepvariant/realigner/BUILD` | Added `@boost//:boost` to `debruijn_graph` |
| `third_party/nucleus/io/BUILD` | Added `@boost//:boost` to `gbz_reader` |
| `deepvariant/fast_pipeline.h` | Boost 1.90 v1 process API includes |
| `deepvariant/fast_pipeline.cc` | Boost 1.90 v1 process API includes |
| `deepvariant/allelecounter.cc` | `int64_t`/`long` type mismatch fix |
| `deepvariant/alt_aligned_pileup_lib.cc` | `int64_t`/`long` type mismatch fixes |
| `deepvariant/make_examples_native.cc` | `int64_t`/`long` type mismatch fix |
| `deepvariant/call_variants.py` | `--use_coreml` / `--coreml_model` flags; CoreML inference path; batch cap at 128; float16 normalization; `num_parallel_calls=1` fix for fast pipeline CPU starvation |
| `deepvariant/make_examples_options.py` | `--hts_num_threads` flag for parallel BAM decompression |
| `deepvariant/make_examples_core.py` | Pass `hts_num_threads` to SAM reader |
| `deepvariant/protos/deepvariant.proto` | `hts_num_threads` field in `MakeExamplesOptions` (tag 94) |
| `third_party/nucleus/protos/reads.proto` | `hts_num_threads` field in `SamReaderOptions` (tag 12) |
| `third_party/nucleus/io/sam_reader.cc` | Call `hts_set_threads()` when `hts_num_threads > 0` |
| `third_party/nucleus/io/sam.py` | Expose `hts_num_threads` through Python SAM reader wrapper |
| `scripts/run_deepvariant.py` | Apple Silicon auto-detection; CoreML auto-enable; fast pipeline auto-enable (`--fast_pipeline` flag); `--batch_size` / `--use_coreml` flags; default shards from perf cores |

### External Patches (Outside This Repo)

| Target | Change |
|--------|--------|
| `tensorflow/tensorflow.bzl` | Replaced GNU `ln -r -s` with Python `os.path.relpath` for macOS |
| `external/zlib/zutil.h` (Bazel cache) | Prevent `fdopen=NULL` on macOS (`TARGET_OS_MAC` guard) |

---

## Known Issues

1. **`postprocess_variants` segfaults with `--cpus >1`.** When `call_variants` detects a GPU, it auto-creates multiple output shards (one per CPU writer thread). If `postprocess_variants` then runs with `--cpus >1`, it partitions the genome and some partitions produce empty VCFs that `bcftools naive_concat` cannot parse, causing a segfault. **Workaround:** use `--cpus 1`. This forces single-partition mode and bypasses the concat. The quicktest and benchmark scripts already apply this fix. The performance impact is negligible for chr20 (~16s) and minor for full genomes (~7-10 minutes single-threaded vs ~2-3 minutes multi-threaded).

2. **`make_examples` appends to existing tfrecord files.** If a previous run was interrupted, leftover partial files will corrupt the next run with `DataLossError: inflate() failed with error -3`. Always clean the output directory before re-running, or use the benchmark script which does this automatically.

3. **zlib `zutil.h` patch is not persistent.** It lives in the Bazel cache and must be reapplied after `bazel clean`. See step 3 above.

4. **zsh escapes `!` in strings.** The build script uses `bytes([0x23, 0x21])` to write `#!` shebangs. If you write custom scripts that generate shebangs, be aware of this.

5. **`int64_t` is `long long` on macOS ARM64**, while it is `long` on Linux x86_64. Both are 64-bit, but they are different types to the compiler, causing `std::max(int64_t, 0L)` template deduction failures. All instances in DeepVariant have been fixed.

6. **Boost 1.90+** split the `process` API into v1 and v2 namespaces. The build uses `-DBOOST_PROCESS_VERSION=1` to select v1.

7. **`tf-models-official`** depends on `tensorflow-text`, which has no ARM64 wheels for 2.13.x. It is installed with `--no-deps`. DeepVariant only uses `official.modeling.optimization`, which does not require `tensorflow-text`.

8. **`CUDA_VISIBLE_DEVICES="-1"` does not disable Metal GPU.** The TensorFlow Metal plugin ignores CUDA environment variables entirely. There is no known way to force CPU-only inference on Apple Silicon when `tensorflow-metal` is installed. To benchmark CPU-only performance, uninstall `tensorflow-metal` from the Python environment.

9. **Fast pipeline stale semaphores.** If `fast_pipeline` is killed mid-run (e.g. Ctrl-C, zombie process), POSIX named semaphores are left locked. Restarting with the same `--shm_prefix` causes `make_examples` to block forever in `StartStreaming()`. Fix: unlink the semaphores before restarting:
   ```python
   import ctypes, ctypes.util
   libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)
   for shard in range(8):  # adjust for your --num_shards value
       for kind in ["buffer_empty", "items_available", "shard_finished"]:
           libc.sem_unlink(f"/dv_bm_1_{kind}_{shard}".encode())
   ```
   The `benchmark.sh` script unlinks stale semaphores automatically before each run.

---

## Running DeepVariant

Once built, DeepVariant is run the same way as on Linux, but using the zip binaries directly instead of Docker:

```bash
# Activate venv with tensorflow-macos
source ~/dv-venv/bin/activate

INPUT_DIR="path/to/input"
OUTPUT_DIR="path/to/output"

# Step 1: make_examples
python3 bazel-bin/deepvariant/make_examples.zip \
  --mode calling \
  --ref "${INPUT_DIR}/reference.fasta" \
  --reads "${INPUT_DIR}/reads.bam" \
  --output "${OUTPUT_DIR}/examples.tfrecord.gz" \
  --examples "${OUTPUT_DIR}/examples.tfrecord.gz"

# Step 2: call_variants (Metal GPU provides ~4x speedup)
python3 bazel-bin/deepvariant/call_variants.zip \
  --outfile "${OUTPUT_DIR}/call_variants_output.tfrecord.gz" \
  --examples "${OUTPUT_DIR}/examples.tfrecord.gz" \
  --checkpoint "path/to/model.ckpt"

# Step 3: postprocess_variants
python3 bazel-bin/deepvariant/postprocess_variants.zip \
  --ref "${INPUT_DIR}/reference.fasta" \
  --infile "${OUTPUT_DIR}/call_variants_output.tfrecord.gz" \
  --outfile "${OUTPUT_DIR}/output.vcf.gz"
```

For the full command reference and model types (WGS, WES, PACBIO, ONT, etc.), see the [upstream documentation](https://github.com/google/deepvariant/blob/r1.9/docs/deepvariant-details.md).

---

## System Requirements

| Component | Requirement |
|-----------|-------------|
| OS | macOS 11+ (Big Sur or later) |
| Architecture | Apple Silicon (M1, M2, M3, M4) |
| Python | 3.10 |
| Bazel | 5.3.0 (installed by `build-prereq-macos.sh`) |
| TensorFlow | 2.13.1 source (build time), `tensorflow-macos` 2.13.1 (runtime) |
| GPU | Metal via `tensorflow-metal` 1.0.0 (4.25x speedup for call_variants) |
| Disk | ~30 GB (TF source + Bazel cache) |
| RAM | 16 GB minimum, 32 GB+ recommended |

---

## How to Cite

If you use DeepVariant in your work, please cite:

[A universal SNP and small-indel variant caller using deep neural networks. *Nature Biotechnology* 36, 983-987 (2018).](https://rdcu.be/7Dhl)
Ryan Poplin, Pi-Chuan Chang, David Alexander, Scott Schwartz, Thomas Colthurst, Alexander Ku, Dan Newburger, Jojo Dijamco, Nam Nguyen, Pegah T. Afshar, Sam S. Gross, Lizzie Dorfman, Cory Y. McLean, and Mark A. DePristo.
doi: https://doi.org/10.1038/nbt.4235

## License

[BSD-3-Clause license](LICENSE)

## Disclaimer

This is not an official Google product.

NOTE: the content of this research code repository (i) is not intended to be a medical device; and (ii) is not intended for clinical use of any kind, including but not limited to diagnosis or prognosis.
