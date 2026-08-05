# ri_kernels

SIMD and GPU kernels for radio interferometry, exposed to JAX as FFI custom
calls.

## Installation

The CPU kernels and all of the Python code live in `ri_kernels`:

```bash
pip install ri_kernels
```

The GPU kernels ship as add-on packages, one per CUDA major version. Pick the
one matching your `jaxlib`; it pulls in `ri_kernels` itself:

```bash
pip install ri_kernels_cuda12   # for jax[cuda12]
pip install ri_kernels_cuda13   # for jax[cuda13]
```

Nothing else changes at the call site — `ri_kernels.jax_api` finds the GPU
library wherever it was installed and registers the `CUDA` FFI targets. If the
library for a platform is not installed, only the lowerings for that platform
raise; the import still succeeds.

There are no ROCm wheels; build from source with `RI_KERNELS_ROCM=1`.

## Building from source

Requires CMake >= 3.20, a C++20 compiler, and network access at configure time
(Google Highway is fetched by `FetchContent`). `jax==0.6.0` is pulled into the
build environment automatically for its XLA FFI headers.

```bash
pip install .                      # CPU only
RI_KERNELS_CUDA=1 pip install .    # CPU + CUDA, needs nvcc
RI_KERNELS_ROCM=1 pip install .    # CPU + ROCm, needs hipcc
```

Both libraries land in the `ri_kernels` package directory in that case, which is
also a location the loader searches.

Other CMake options of note: `RI_KERNELS_CPU` (default `ON`),
`RI_KERNELS_MULTI_ARCH` (dynamic SIMD dispatch, default `ON` — turn it off and
set arch flags via `CMAKE_CXX_FLAGS` for a single-target build),
`RI_KERNELS_BUNDLED_HIGHWAY`, and `CMAKE_CUDA_ARCHITECTURES` (default
`80;90a-real;90`).

## Tests

```bash
pytest tests
```

The tests skip themselves when no kernel library is importable, so make sure
`ri_kernels` is installed (or built in place) first.

## Releasing

`.github/workflows/wheels.yml` builds all three distributions on every push and
publishes them to PyPI when a GitHub Release is published. The release tag must
match `[project].version` in `pyproject.toml`.

The CUDA wheels are generated from the same source tree by `ci/make_variant.py`,
which rewrites `pyproject.toml` into the add-on's metadata; there is only one
place to bump the version.
