# CUDA Batch Grayscale Converter

A GPU-accelerated batch image processing tool that converts RGB images to grayscale using **CUDA** and the **NVIDIA NPP (NVIDIA Performance Primitives)** library.

---

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Building](#building)
- [Running](#running)
- [Command-Line Arguments](#command-line-arguments)
- [Output](#output)
- [Algorithm / Kernels Used](#algorithm--kernels-used)
- [Lessons Learned](#lessons-learned)

---

## Overview

This project processes a large batch of images entirely on the GPU.  
For each input image the pipeline is:

```
Load image (CPU) → Copy to GPU → nppiRGBToGray_8u_C3C1R (GPU) → Copy back → Save (CPU)
```

Using the NPP library's `nppiRGBToGray_8u_C3C1R` kernel ensures the heavy pixel-math runs in parallel across thousands of CUDA cores.

---

## Requirements

| Dependency | Version |
|---|---|
| CUDA Toolkit | ≥ 11.0 |
| NVIDIA Driver | Compatible with your CUDA version |
| NPP Library | Included with CUDA Toolkit |
| g++ / nvcc | C++17 support |
| Python 3 | Optional — for downloading sample images |

---

## Project Structure

```
.
├── main.cu                  # Main CUDA source file
├── Makefile                 # Build system
├── run.sh                   # One-command build + run script
├── download_images.py       # Helper to download test images
├── include/
│   ├── stb_image.h          # Image loading (header-only)
│   └── stb_image_write.h    # Image saving (header-only)
├── data/
│   └── images/              # Put your input images here
└── output/                  # Grayscale results saved here (auto-created)
```

---

## Setup

### 1. Clone this repository

```bash
git clone https://github.com/YOUR_USERNAME/cuda-batch-grayscale.git
cd cuda-batch-grayscale
```

### 2. Download the stb header-only libraries

```bash
mkdir -p include
curl -o include/stb_image.h \
     https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
curl -o include/stb_image_write.h \
     https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h
```

### 3. Add test images

**Option A — Use the download script:**
```bash
python3 download_images.py --count 50 --output data/images
```

**Option B — Copy your own images:**
```bash
mkdir -p data/images
cp /path/to/your/images/*.jpg data/images/
```

---

## Building

```bash
make
```

> **Note:** The Makefile defaults to `sm_75` (Turing GPUs like RTX 2000 series).  
> Change `-arch=sm_75` in the Makefile to match your GPU:
> - `sm_60` — Pascal (GTX 10xx)
> - `sm_70` — Volta (V100)
> - `sm_75` — Turing (RTX 20xx)
> - `sm_80` — Ampere (RTX 30xx / A100)
> - `sm_89` — Ada Lovelace (RTX 40xx)

---

## Running

### Quick start (uses `data/images` as input, saves to `output/`):

```bash
./run.sh
```

### Manual run with custom paths:

```bash
./grayscale_converter --input data/images --output output --log output/run.log
```

---

## Command-Line Arguments

| Argument | Required | Description |
|---|---|---|
| `--input <dir>` | ✅ Yes | Directory containing input `.jpg`/`.png` images |
| `--output <dir>` | ✅ Yes | Directory where grayscale output images are saved |
| `--log <file>` | ❌ No | Path to log file (defaults to stdout) |
| `--help` / `-h` | ❌ No | Show usage information |

---

## Output

Each input image `image_0001.jpg` produces `image_0001_gray.png` in the output directory.

A log file `run.log` is written with:
- GPU device name
- Per-image processing time (ms)
- Summary of total images processed, successes, failures, and total time

Example log output:
```
=== CUDA Batch Grayscale Converter ===
GPU: NVIDIA GeForce RTX 3080
CUDA devices found: 1

Found 50 image(s) in: data/images

Processing: image_0001.jpg -> image_0001_gray.png  [OK]  3.21 ms
Processing: image_0002.jpg -> image_0002_gray.png  [OK]  2.98 ms
...

=== Summary ===
Total images processed : 50
Successful             : 50
Failed                 : 0
Total wall-clock time  : 187.44 ms
Output directory       : output
```

---

## Algorithm / Kernels Used

### `nppiRGBToGray_8u_C3C1R` (NVIDIA NPP)

Converts a 3-channel 8-bit RGB image to a 1-channel 8-bit grayscale image using the standard luminance formula:

```
Gray = 0.299·R + 0.587·G + 0.114·B
```

This computation runs in parallel on the GPU across all pixels simultaneously, making it ideal for large batches of high-resolution images.

---

## Lessons Learned

- The NVIDIA NPP library makes GPU image processing very accessible — a single function call replaces hundreds of lines of custom kernel code.
- Memory transfer (CPU ↔ GPU) is often the bottleneck for small images; batching transfers or using pinned memory could further improve performance.
- Using `stb_image` for I/O keeps the project dependency-free beyond CUDA/NPP.
- The `sm_XX` compute capability flag in the Makefile must match the target GPU — mismatches cause silent runtime errors.

---

## License

This project is submitted as part of the GPU Specialization Capstone on Coursera.  
`stb_image` and `stb_image_write` are public domain (Sean Barrett).
