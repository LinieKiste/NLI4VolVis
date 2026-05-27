# Build Modifications

This document describes the changes required to build NLI4VolVis on a system with a modern GPU (RTX 5060 Ti / Blackwell, sm_120), GCC 16, and glibc newer than what CUDA 12.1 was designed for.

## Environment

- **OS**: Arch Linux
- **GPU**: NVIDIA GeForce RTX 5060 Ti (Blackwell, sm_120)
- **System GCC**: 16.1.1 (too new for any CUDA 12.x)
- **conda env**: `nli4volvis` (Python 3.9)

---

## 1. Upgraded CUDA toolkit (12.1 → 12.8)

CUDA 12.1 does not support Blackwell (sm_120). Upgraded via conda:

```bash
conda install -n nli4volvis -c "nvidia/label/cuda-12.8.0" cuda-toolkit=12.8.0
```

## 2. Upgraded PyTorch (2.1.2+cu121 → 2.8.0+cu128)

PyTorch 2.1.2 does not support Blackwell. Upgraded to match CUDA 12.8:

```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## 3. Installed GCC 12 system-wide

CUDA 12.8's nvcc only supports up to GCC 14, and the system default (GCC 16) causes errors. GCC 12 was installed via AUR:

```bash
yay -S gcc12
```

GCC 12 binaries land at `/usr/bin/gcc-12` and `/usr/bin/g++-12`.

## 4. Patched CUDA `host_config.h` (GCC version limit)

CUDA's header enforces a hard GCC version check. Raised the limit so GCC 16 doesn't block compilation (nvcc still uses gcc-12 explicitly, but this avoids errors if the system gcc is picked up):

**File**: `$CONDA_PREFIX/include/crt/host_config.h`

```diff
-#if __GNUC__ > 12
+#if __GNUC__ > 20
```

## 5. Patched CUDA `math_functions.h` (glibc noexcept conflict)

Modern glibc declares `cospi`, `sinpi`, and `rsqrt` (and their float variants) with `noexcept(true)`, but CUDA's headers declared them without `noexcept`, causing a redeclaration error.

**File**: `$CONDA_PREFIX/targets/x86_64-linux/include/crt/math_functions.h`

Added `noexcept` to 6 declarations:

```diff
-extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double  rsqrt(double x);
+extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double  rsqrt(double x) noexcept;

-extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float   rsqrtf(float x);
+extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float   rsqrtf(float x) noexcept;

-extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double  sinpi(double x);
+extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double  sinpi(double x) noexcept;

-extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float   sinpif(float x);
+extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float   sinpif(float x) noexcept;

-extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double  cospi(double x);
+extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double  cospi(double x) noexcept;

-extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float   cospif(float x);
+extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float   cospif(float x) noexcept;
```

## 6. Fixed missing `#include <float.h>` in simple-knn

`simple_knn.cu` uses `FLT_MAX` but did not include the header that defines it.

**File**: `submodules/simple-knn/simple_knn.cu`

```diff
 #include "cuda_runtime.h"
 #include "device_launch_parameters.h"
+#include <float.h>
```

## 7. Created missing `__init__.py` for simple-knn

The `simple_knn/` directory had no `__init__.py`, making the package unimportable after editable install.

```bash
touch submodules/simple-knn/simple_knn/__init__.py
```

---

## Build commands for the CUDA extensions

Always set these environment variables before building:

```bash
export CUDA_HOME=/home/casimir/.conda/envs/nli4volvis
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export TORCH_CUDA_ARCH_LIST='7.5;8.0;8.6;9.0;12.0'
```

Then:

```bash
conda activate nli4volvis
pip install -e ./submodules/diff-gaussian-rasterization --no-build-isolation
pip install -e ./submodules/simple-knn --no-build-isolation
```

## Downgraded NumPy

NumPy 2.x was pre-installed and caused build issues. Downgraded to 1.x:

```bash
pip install "numpy<2"
```
