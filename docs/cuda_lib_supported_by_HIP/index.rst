CUDA libraries supported by HIP 
===============================

Library Equivalents
-------------------
See the table below for library equivalents between CUDA and ROCm.

.. todo: no more ROCm anywhere

+-----------------------+-----------------------------+----------------+
| CUDA Library          | ROCm Library                | Comment        |
+=======================+=============================+================+
| cuBLAS                | rocBLAS                     | Basic Linear   |
|                       |                             | Algebra        |
|                       |                             | Subroutines    |
+-----------------------+-----------------------------+----------------+
| cuFFT                 | rocFFT                      | Fast Fourier   |
|                       |                             | Transfer       |
|                       |                             | Library        |
+-----------------------+-----------------------------+----------------+
| cuSPARSE              | rocSPARSE                   | Sparse BLAS +  |
|                       |                             | SPMV           |
+-----------------------+-----------------------------+----------------+
| cuSolver              | rocSOLVER                   | Lapack library |
+-----------------------+-----------------------------+----------------+
| AMG-X                 | rocALUTION                  | Sparse         |
|                       |                             | iterative      |
|                       |                             | solvers and    |
|                       |                             | p              |
|                       |                             | reconditioners |
|                       |                             | with Geometric |
|                       |                             | and Algebraic  |
|                       |                             | MultiGrid      |
+-----------------------+-----------------------------+----------------+
| Thrust                | rocThrust                   | C++ parallel   |
|                       |                             | algorithms     |
|                       |                             | library        |
+-----------------------+-----------------------------+----------------+
| CUB                   | rocPRIM                     | Low Level      |
|                       |                             | Optimized      |
|                       |                             | Parallel       |
|                       |                             | Primitives     |
+-----------------------+-----------------------------+----------------+
| cuDNN                 | MIOpen                      | Deep learning  |
|                       |                             | Solver Library |
+-----------------------+-----------------------------+----------------+
| cuRAND                | rocRAND                     | Random Number  |
|                       |                             | Generator      |
|                       |                             | Library        |
+-----------------------+-----------------------------+----------------+
| EIGEN                 | EIGEN HIP port              | C++ template   |
|                       |                             | library for    |
|                       |                             | linear         |
|                       |                             | algebra:       |
|                       |                             | matrices,      |
|                       |                             | vectors,       |
|                       |                             | numerical      |
|                       |                             | solvers,       |
+-----------------------+-----------------------------+----------------+
| NCCL                  | RCCL                        | Communications |
|                       |                             | Primitives     |
|                       |                             | Library based  |
|                       |                             | on the MPI     |
|                       |                             | equivalents    |
+-----------------------+-----------------------------+----------------+


API of the above mentioned libraries:

.. toctree::
    :maxdepth: 2

    CUBLAS_API_supported_by_HIP.md
    cuComplex_API_supported_by_HIP.md
    CUDNN_API_supported_by_HIP.md
    CUFFT_API_supported_by_HIP.md
    CURAND_API_supported_by_HIP.md
    CUSPARSE_API_supported_by_HIP.md