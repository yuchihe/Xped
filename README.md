[![License](https://img.shields.io/static/v1?label=LICENSE&message=GPL&color=green&style=for-the-badge)](./LICENSE)
[![Documentation](https://img.shields.io/static/v1?label=DOCS&message=dev&color=blue&style=for-the-badge)](https://cpp977.github.io/Xped/)
![Linux](https://img.shields.io/static/v1?label=OS&message=LINUX&color=orange&style=for-the-badge)
![Windows](https://img.shields.io/static/v1?label=OS&message=WINDOWS&color=orange&style=for-the-badge)

## Xped
Library for the manipulation of symmetric (block-sparse) tensors with arbitrary (**X**) amount of indices (**ped**).

## Status

|Builds  | Tests | 	Coverage |
|:-: | :-: | :-: |
| [![Builds](https://github.com/cpp977/Xped/workflows/Builds/badge.svg)](https://github.com/cpp977/Xped/actions)|[![Tests](https://github.com/cpp977/Xped/workflows/Tests/badge.svg)](https://github.com/cpp977/Xped/actions)|[![codecov](https://codecov.io/gh/cpp977/Xped/branch/master/graph/badge.svg?token=MRQLD834VO)](https://codecov.io/gh/cpp977/Xped)|

## Getting started

https://github.com/cpp977/Xped/blob/1eab6bf97de1fb84b3992361436996391a4f0292/docs/snippets/quickstart.cpp#L1-L47

## Build

1. Get the sources using:
`git clone --recurse-submodules https://github.com/cpp977/Xped`
3. Configure with cmake:
`cmake --preset=<preset> /path/to/source/Xped`
This also installs dependencies via vcpkg so the first run takes several minutes.
The build directory is specified in the presets: `/path/to/source/Xped/../<preset-name>`
4. Build tests:
`cmake --build --preset=<preset>`
5. Run the tests:
`ctest --preset=<preset>`

To control the build, it is recomennded to choose a CMake preset so that several options are already set automatically.
The available presets follow the scheme `<compiler>-<backend>-<build-type>`. E.g. `gcc-eigen-release` uses the gnu c++ compiler and the Eigen backend and performs a release build.
Supported compilers are gcc (version >= 10), clang (version >= 12), msvc (version >= 19.30) and intel icpx (version >= 2021.04).

For the Eigen backend, we recommend Intel MKL (`-DXPED_USE_MKL=ON`) for BLAS and LAPACK.
The recommended SVD driver is LAPACK's divide-and-conquer `gesdd` (`dgesdd` for real double precision), selected through Eigen's `BDCSVD` when LAPACKE is enabled.
The legacy `gesvd` path remains available by defining `XPED_DONT_USE_BDCSVD` at compile time.

All build options can be seen in the following table.

### Build Options

| Option | Default | Description |
| --- | --- | --- |
| `XPED_BUILD_BENCHMARKS` | `ON` | Build the benchmarks. |
| `XPED_BUILD_CYCLOPS` | `OFF` | Build the cyclops library from source. |
| `XPED_BUILD_EXAMPLES` | `OFF` | Build the benchmarks. |
| `XPED_BUILD_TESTS` | `ON` | Build the tests. |
| `XPED_BUILD_TOOLS` | `ON` | Build the tools. |
| `XPED_COMPILED_LIB` | `OFF` | Configure the library as a compiled library. Long compile times. |
| `XPED_EFFICIENCY_MODEL` | `XPED_TIME_EFFICIENT` | Xped tries to be time efficient. |
| `XPED_ENABLE_BUILD_WITH_TIME_TRACE` | `OFF` | Enable -ftime-trace to generate time tracing .json files on clang |
| `XPED_ENABLE_CCACHE` | `OFF` | Enable a compiler cache if available |
| `XPED_ENABLE_CLANG_FORMAT` | `ON` | Enable clang-format target. |
| `XPED_ENABLE_CLANG_TIDY` | `OFF` | Enable static analysis with clang-tidy |
| `XPED_ENABLE_COVERAGE` | `OFF` | Enable coverage reporting for gcc/clang |
| `XPED_ENABLE_CPPCHECK` | `OFF` | Enable static analysis with cppcheck |
| `XPED_ENABLE_DOXYGEN` | `OFF` | Enable doxygen doc builds of source |
| `XPED_ENABLE_INCLUDE_WHAT_YOU_USE` | `OFF` | Enable static analysis with include-what-you-use |
| `XPED_ENABLE_IPO` | `OFF` | Enable Interprocedural Optimization, aka Link Time Optimization (LTO) |
| `XPED_ENABLE_LRU_CACHE` | `OFF` | Use lru cache library from github. |
| `XPED_ENABLE_SANITIZER_ADDRESS` | `OFF` | Enable address sanitizer |
| `XPED_ENABLE_SANITIZER_LEAK` | `OFF` | Enable leak sanitizer |
| `XPED_ENABLE_SANITIZER_MEMORY` | `OFF` | Enable memory sanitizer |
| `XPED_ENABLE_SANITIZER_THREAD` | `OFF` | Enable thread sanitizer |
| `XPED_ENABLE_SANITIZER_UNDEFINED_BEHAVIOR` | `OFF` | Enable undefined behavior sanitizer |
| `XPED_LAPACKE` | `/usr/lib/x86_64-linux-gnu/liblapacke.so` | Path to a library. |
| `XPED_LOG_LEVEL` | `SPDLOG_LEVEL_CRITICAL` | Compile time log level. |
| `XPED_MATRIX_LIB` | `Eigen` | Used matrix library for plain tensor operations. |
| `XPED_OPTIM_LIB` | `ceres` | Used library for nonlinear gradient-based optimization. |
| `XPED_PEDANTIC_ASSERTS` | `OFF` | Enables rigorous assertions for tensor operations. |
| `XPED_STORAGE` | `Contiguous` | Used storage for Xped::Tensor. |
| `XPED_TENSOR_LIB` | `Eigen` | Used tensor library for plain tensor operations. |
| `XPED_USE_AD` | `ON` | Use automatic differentiation (AD) with Xped Tensors. |
| `XPED_USE_EXTERNAL_STAN` | `OFF` | Use the external stan-math library for AD instead of the bundled self-contained engine. |
| `XPED_USE_EXTERNAL_CERES` | `OFF` | Use the external ceres-solver instead of the bundled self-contained L-BFGS engine. |
| `XPED_USE_GSL_COEFFS` | `OFF` | Use GSL for SU(2) coupling coefficients instead of the bundled self-contained implementation. |
| `XPED_USE_BLAS` | `ON` | Enable blas linking. |
| `XPED_USE_LAPACK` | `ON` | Enable lapack linking. |
| `XPED_USE_LIBCXX` | `OFF` | Use libc++ from llvm. |
| `XPED_USE_MKL` | `OFF` | Enable use of intel math kernel library (MKL). |
| `XPED_USE_MPI` | `OFF` | Enable message parsing interface (mpi) parallelization |
| `XPED_USE_NLO` | `ON` | Use nonlinear optimization algorithms. |
| `XPED_USE_OPENMP` | `ON` | Enable openmp parallelization |
| `XPED_USE_SCALAPACK` | `OFF` | Enable scalapack linking (only useful for MPI programs). |
| `XPED_VECTOR_LIB` | `Eigen` | Used vector library for plain tensor operations. |

## PEPS energy evaluation

The Eigen backend evaluates real double-precision U0 two-site observables with exact slicing for horizontal/vertical nearest-neighbor bonds and both next-nearest-neighbor diagonals (`TRank=2`, `ALL_OUT_LEGS=false`).
`avg` and `CTMSolver` select this path automatically; `avg_sliced(env, op, block_size=8)` exposes the boundary block width.
Interaction and physical-index contributions are evaluated in turn, and native AD replays each contribution while accumulating tensor and boundary gradients, including the normalization derivative.
The complete operator is retained, including channels that the historical diagonal observable cutoff could discard.
See [exact energy slicing](docs/main.md#exact-energy-slicing) for the untruncated reference, contraction geometry, memory limits, and benchmark scope.

# Documentation

Browse the [documenation](https://cpp977.github.io/Xped "docs") to learn about the capabilities of the library.

## Acknowledgment

This library is inspired by a tensor framework for symmetric tensors written in the Julia programming language: [TensorKit](https://github.com/Jutho/TensorKit.jl).

