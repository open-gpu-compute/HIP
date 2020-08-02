Compiler Directives
===================

Distinguishing Compiler Modes
-----------------------------

Identifying HIP Target Platform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All HIP projects target either AMD or NVIDIA platform. The platform affects
which headers are included and which libraries are used for linking.

-  ``HIP_PLATFORM_HCC`` is defined if the HIP platform targets AMD

-  ``HIP_PLATFORM_NVCC`` is defined if the HIP platform targets NVIDIA

On AMD platform, the compiler was hcc, but is deprecated in ROCM v3.5 release,
and HIP-Clang compiler is introduced for compiling HIP programs.

For most HIP applications, the transition from hcc to HIP-Clang is
transparent. HIPCC and HIP cmake files automatically choose compilation
options for HIP-Clang and hide the difference between the hcc and hip-clang
code. However, minor changes may be required as HIP-Clang has stricter syntax
and semantic checks compared to hcc.

Many projects use a mixture of an accelerator compiler (AMD or NVIDIA) and a
standard compiler (e.g. g++). These defines are set for both accelerator and
standard compilers and thus are often the best option when writing code that
uses conditional compilation.

Identifying the Compiler: hcc, hip-clang or nvcc
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Often, it is useful to know whether the underlying compiler is hcc, HIP-Clang
or nvcc. This knowledge can guard platform-specific code or aid in
platform-specific performance tuning.

.. code-block:: cpp

   #ifdef __HCC__
   // Compiled with hcc

   #ifdef __HIP__
   // Compiled with HIP-Clang

   #ifdef __NVCC__
   // Compiled with nvcc
   // Could be compiling with CUDA language extensions enabled (for example, a ".cu file)
   // Could be in pass-through mode to an underlying host compile OR (for example, a .cpp file)

   #ifdef __CUDACC__
   // Compiled with nvcc (CUDA language extensions enabled)

Compiler directly generates the host code (using the Clang x86 target) and
passes the code to another host compiler. Thus, they have no equivalent of the
``__CUDA_ACC`` define.

Identifying Current Compilation Pass: Host or Device
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

nvcc makes two passes over the code: one for host code and one for device code. HIP-Clang will have multiple passes over the code: one for the host code, and one for each architecture on the device code.
``__HIP_DEVICE_COMPILE__`` is set to a nonzero value when the compiler
(hcc, HIP-Clang or nvcc) is compiling code for a device inside a
``__global__`` kernel or for a device function.
``__HIP_DEVICE_COMPILE__`` can replace ``#ifdef`` checks on the
``__CUDA_ARCH__`` define.

.. code-block:: cpp

   // #ifdef __CUDA_ARCH__
   #if __HIP_DEVICE_COMPILE__

Unlike ``__CUDA_ARCH__``, the ``__HIP_DEVICE_COMPILE__`` value is 1 or
undefined, and it doesn't represent the feature capability of the target
device.

Compiler Defines: Summary
~~~~~~~~~~~~~~~~~~~~~~~~~
The table below provides a summary of compiler defines.

.. todo: make this table beautiful


+-------------+-------------+-------------+-------------+-------------+
| Define      | hcc         | HIP-Clang   | nvcc        | Other (GCC, |
|             |             |             |             | ICC, Clang, |
|             |             |             |             | etc.)       |
+=============+=============+=============+=============+=============+
| HIP-related |             |             |             |             |
| defines:    |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| ``          | Defined     | Defined     | Undefined   | Defined if  |
| __HIP_PLATF |             |             |             | targeting   |
| ORM_HCC__`` |             |             |             | hcc         |
|             |             |             |             | platform;   |
|             |             |             |             | undefined   |
|             |             |             |             | otherwise   |
+-------------+-------------+-------------+-------------+-------------+
| ``_         | Undefined   | Undefined   | Defined     | Defined if  |
| _HIP_PLATFO |             |             |             | targeting   |
| RM_NVCC__`` |             |             |             | nvcc        |
|             |             |             |             | platform;   |
|             |             |             |             | undefined   |
|             |             |             |             | otherwise   |
+-------------+-------------+-------------+-------------+-------------+
| ``__        | 1 if        | 1 if        | 1 if        | Undefined   |
| HIP_DEVICE_ | compiling   | compiling   | compiling   |             |
| COMPILE__`` | for device; | for device; | for device; |             |
|             | undefined   | undefined   | undefined   |             |
|             | if          | if          | if          |             |
|             | compiling   | compiling   | compiling   |             |
|             | for host    | for host    | for host    |             |
+-------------+-------------+-------------+-------------+-------------+
| ``          | Defined     | Defined     | Defined     | Undefined   |
| __HIPCC__`` |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| ``__H       | 0 or 1      | 0 or 1      | 0 or 1      | 0           |
| IP_ARCH_*`` | depending   | depending   | depending   |             |
|             | on feature  | on feature  | on feature  |             |
|             | support     | support     | support     |             |
|             | (see below) | (see below) | (see below) |             |
+-------------+-------------+-------------+-------------+-------------+
| n           |             |             |             |             |
| vcc-related |             |             |             |             |
| defines:    |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| ``_         | Undefined   | Undefined   | Defined if  | Undefined   |
| _CUDACC__`` |             |             | source code |             |
|             |             |             | is compiled |             |
|             |             |             | by nvcc;    |             |
|             |             |             | undefined   |             |
|             |             |             | otherwise   |             |
+-------------+-------------+-------------+-------------+-------------+
| `           | Undefined   | Undefined   | Defined     | Undefined   |
| `__NVCC__`` |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| ``__CU      | Undefined   | Undefined   | Unsigned    | Undefined   |
| DA_ARCH__`` |             |             | r           |             |
|             |             |             | epresenting |             |
|             |             |             | compute     |             |
|             |             |             | capability  |             |
|             |             |             | For example,|             |
|             |             |             | if          |             |
|             |             |             | in device   |             |
|             |             |             | code; 0 if  |             |
|             |             |             | in host     |             |
|             |             |             | code        |             |
+-------------+-------------+-------------+-------------+-------------+
| hcc-related |             |             |             |             |
| defines:    |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| ``__HCC__`` | Defined     | Undefined   | Undefined   | Undefined   |
+-------------+-------------+-------------+-------------+-------------+
| `           | Nonzero if  | Undefined   | Undefined   | Undefined   |
| `__HCC_ACCE | in device   |             |             |             |
| LERATOR__`` | code;       |             |             |             |
|             | otherwise   |             |             |             |
|             | undefined   |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| hip-cl      |             |             |             |             |
| ang-related |             |             |             |             |
| defines:    |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| ``__HIP__`` | Undefined   | Defined     | Undefined   | Undefined   |
+-------------+-------------+-------------+-------------+-------------+
| hc          |             |             |             |             |
| c/HIP-Clang |             |             |             |             |
| common      |             |             |             |             |
| defines:    |             |             |             |             |
+-------------+-------------+-------------+-------------+-------------+
| ``          | Defined     | Defined     | Undefined   | Defined if  |
| __clang__`` |             |             |             | using       |
|             |             |             |             | Clang;      |
|             |             |             |             | otherwise   |
|             |             |             |             | undefined   |
+-------------+-------------+-------------+-------------+-------------+



Identifying Architecture Features
---------------------------------

HIP_ARCH Defines
~~~~~~~~~~~~~~~~

Some CUDA code tests ``__CUDA_ARCH__`` for a specific value to determine whether the machine supports a certain architectural feature. For instance,

::

   #if (__CUDA_ARCH__ >= 130)
   // doubles are supported

This type of code requires special attention, since hcc/AMD and nvcc/CUDA devices have different architectural capabilities. Moreover,
you cannot determine the presence of a feature using a simple comparison against an architecture's version number. HIP provides a set of defines and device properties to query whether a specific architectural feature is supported.

The ``__HIP_ARCH_*`` defines can replace comparisons of
``__CUDA_ARCH__`` values:

::

   //#if (__CUDA_ARCH__ >= 130)   // non-portable
   if __HIP_ARCH_HAS_DOUBLES__ {  // portable HIP feature query
      // doubles are supported
   }

For host code, the ``__HIP_ARCH__*`` defines are set to 0. You should
only use the **HIP_ARCH** fields in device code.

Device-Architecture Properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Host code should query the architecture feature flags in the device properties that hipGetDeviceProperties returns, rather than testing the "major" and "minor" fields directly:

::

   hipGetDeviceProperties(&deviceProp, device);
   //if ((deviceProp.major == 1 && deviceProp.minor < 2))  // non-portable
   if (deviceProp.arch.hasSharedInt32Atomics) {            // portable HIP feature query
       // has shared int32 atomic operations ...
   }

Table of Architecture Properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The table below shows the full set of architectural properties that HIP supports.

+-----------------------+-----------------------------+----------------+
| Define (use only in   | Device Property (run-time   | Comment        |
| device code)          | query)                      |                |
+=======================+=============================+================+
| 32-bit atomics:       |                             |                |
+-----------------------+-----------------------------+----------------+
| ``__HIP_ARCH_HAS_GLO  | hasGlobalInt32Atomics       | 32-bit integer |
| BAL_INT32_ATOMICS__`` |                             | atomics for    |
|                       |                             | global memory  |
+-----------------------+-----------------------------+----------------+
| ``_                   | hasGlobalFloatAtomicExch    | 32-bit float   |
| _HIP_ARCH_HAS_GLOBAL_ |                             | atomic         |
| FLOAT_ATOMIC_EXCH__`` |                             | exchange for   |
|                       |                             | global memory  |
+-----------------------+-----------------------------+----------------+
| ``__HIP_ARCH_HAS_SHA  | hasSharedInt32Atomics       | 32-bit integer |
| RED_INT32_ATOMICS__`` |                             | atomics for    |
|                       |                             | shared memory  |
+-----------------------+-----------------------------+----------------+
| ``_                   | hasSharedFloatAtomicExch    | 32-bit float   |
| _HIP_ARCH_HAS_SHARED_ |                             | atomic         |
| FLOAT_ATOMIC_EXCH__`` |                             | exchange for   |
|                       |                             | shared memory  |
+-----------------------+-----------------------------+----------------+
| ``__HIP_ARCH_HAS      | hasFloatAtomicAdd           | 32-bit float   |
| _FLOAT_ATOMIC_ADD__`` |                             | atomic add in  |
|                       |                             | global and     |
|                       |                             | shared memory  |
+-----------------------+-----------------------------+----------------+
| 64-bit atomics:       |                             |                |
+-----------------------+-----------------------------+----------------+
| ``__HIP_ARCH_HAS_GLO  | hasGlobalInt64Atomics       | 64-bit integer |
| BAL_INT64_ATOMICS__`` |                             | atomics for    |
|                       |                             | global memory  |
+-----------------------+-----------------------------+----------------+
| ``__HIP_ARCH_HAS_SHA  | hasSharedInt64Atomics       | 64-bit integer |
| RED_INT64_ATOMICS__`` |                             | atomics for    |
|                       |                             | shared memory  |
+-----------------------+-----------------------------+----------------+
| Doubles:              |                             |                |
+-----------------------+-----------------------------+----------------+
| ``__HIP               | hasDoubles                  | Do             |
| _ARCH_HAS_DOUBLES__`` |                             | uble-precision |
|                       |                             | floating point |
+-----------------------+-----------------------------+----------------+
| Warp cross-lane       |                             |                |
| operations:           |                             |                |
+-----------------------+-----------------------------+----------------+
| ``__HIP_A             | hasWarpVote                 | Warp vote      |
| RCH_HAS_WARP_VOTE__`` |                             | instructions   |
|                       |                             | (any, all)     |
+-----------------------+-----------------------------+----------------+
| ``__HIP_ARC           | hasWarpBallot               | Warp ballot    |
| H_HAS_WARP_BALLOT__`` |                             | instructions   |
+-----------------------+-----------------------------+----------------+
| ``__HIP_ARCH          | hasWarpShuffle              | Warp shuffle   |
| _HAS_WARP_SHUFFLE__`` |                             | operations     |
|                       |                             | (shfl_*)       |
+-----------------------+-----------------------------+----------------+
| ``__HIP_ARCH_HAS_     | hasFunnelShift              | Funnel shift   |
| WARP_FUNNEL_SHIFT__`` |                             | two input      |
|                       |                             | words into one |
+-----------------------+-----------------------------+----------------+
| Sync:                 |                             |                |
+-----------------------+-----------------------------+----------------+
| ``__HIP_ARCH_HAS_TH   | hasThreadFenceSystem        | thre           |
| READ_FENCE_SYSTEM__`` |                             | adfence_system |
+-----------------------+-----------------------------+----------------+
| ``__HIP_ARCH_HA       | hasSyncThreadsExt           | sync           |
| S_SYNC_THREAD_EXT__`` |                             | threads_count, |
|                       |                             | sy             |
|                       |                             | ncthreads_and, |
|                       |                             | syncthreads_or |
+-----------------------+-----------------------------+----------------+
| Miscellaneous:        |                             |                |
+-----------------------+-----------------------------+----------------+
| ``__HIP_ARCH_         | hasSurfaceFuncs             |                |
| HAS_SURFACE_FUNCS__`` |                             |                |
+-----------------------+-----------------------------+----------------+
| ``__HI                | has3dGrid                   | Grids and      |
| P_ARCH_HAS_3DGRID__`` |                             | groups are 3D  |
+-----------------------+-----------------------------+----------------+
| ``__HIP_ARCH_HAS      | hasDynamicParallelism       |                |
| _DYNAMIC_PARALLEL__`` |                             |                |
+-----------------------+-----------------------------+----------------+
