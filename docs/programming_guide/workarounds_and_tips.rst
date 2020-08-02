Workarounds and Tips
====================

Workarounds
-----------

warpSize
~~~~~~~~

Code should not assume a warp size of 32 or 64. See `Warp Cross-Lane
Functions <hip_kernel_language.md#warp-cross-lane-functions>`__ for
information on how to write portable wave-aware code.

Kernel launch with group size > 256
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Kernel code should use
``__attribute__((amdgpu_flat_work_group_size(<min>,<max>)))``.

For example:

::

   __global__ void dot(double *a,double *b,const int n) __attribute__((amdgpu_flat_work_group_size(1, 512)))

memcpyToSymbol
--------------

HIP support for hipMemcpyToSymbol is complete. This feature allows a
kernel to define a device-side data symbol which can be accessed on the
host side. The symbol can be in \__constant or device space.

Note that the symbol name needs to be encased in the HIP_SYMBOL macro,
as shown in the code example below. This also applies to
hipMemcpyFromSymbol, hipGetSymbolAddress, and hipGetSymbolSize.

For example:

Device Code:

::

   #include<hip/hip_runtime.h>
   #include<hip/hip_runtime_api.h>
   #include<iostream>

   #define HIP_ASSERT(status) \
       assert(status == hipSuccess)

   #define LEN 512
   #define SIZE 2048

   __constant__ int Value[LEN];

   __global__ void Get(hipLaunchParm lp, int *Ad)
   {
       int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
       Ad[tid] = Value[tid];
   }

   int main()
   {
       int *A, *B, *Ad;
       A = new int[LEN];
       B = new int[LEN];
       for(unsigned i=0;i<LEN;i++)
       {
           A[i] = -1*i;
           B[i] = 0;
       }

       HIP_ASSERT(hipMalloc((void**)&Ad, SIZE));

       HIP_ASSERT(hipMemcpyToSymbol(HIP_SYMBOL(Value), A, SIZE, 0, hipMemcpyHostToDevice));
       hipLaunchKernel(Get, dim3(1,1,1), dim3(LEN,1,1), 0, 0, Ad);
       HIP_ASSERT(hipMemcpy(B, Ad, SIZE, hipMemcpyDeviceToHost));

       for(unsigned i=0;i<LEN;i++)
       {
           assert(A[i] == B[i]);
       }
       std::cout<<"Passed"<<std::endl;
   }

threadfence_system
------------------

Threadfence_system makes all device memory writes, all writes to mapped
host memory, and all writes to peer memory visible to CPU and other GPU
devices. Some implementations can provide this behavior by flushing the
GPU L2 cache. HIP/HCC does not provide this functionality. As a
workaround, users can set the environment variable
``HSA_DISABLE_CACHE=1`` to disable the GPU L2 cache. This will affect
all accesses and for all kernels and so may have a performance impact.

Textures and Cache Control
~~~~~~~~~~~~~~~~~~~~~~~~~~

Compute programs sometimes use textures either to access dedicated
texture caches or to use the texture-sampling hardware for interpolation
and clamping. The former approach uses simple point samplers with linear
interpolation, essentially only reading a single point. The latter
approach uses the sampler hardware to interpolate and combine multiple
point samples. AMD hardware, as well as recent competing hardware, has a
unified texture/L1 cache, so it no longer has a dedicated texture cache.
But the nvcc path often caches global loads in the L2 cache, and some
programs may benefit from explicit control of the L1 cache contents. We
recommend the \__ldg instruction for this purpose.

AMD compilers currently load all data into both the L1 and L2 caches, so
\__ldg is treated as a no-op.

We recommend the following for functional portability:

-  For programs that use textures only to benefit from improved caching,
   use the \__ldg instruction
-  Programs that use texture object and reference APIs, work well on HIP

More Tips
---------

HIPTRACE Mode
~~~~~~~~~~~~~~~~~~

On an hcc/AMD platform, set the HIP_TRACE_API environment variable to see a textural API trace. Use the following bit mask:
       
   - 0x1 = trace APIs
   - 0x2 = trace synchronization operations
   - 0x4 = trace memory allocation / deallocation

Environment Variables
~~~~~~~~~~~~~~~~~~~~~~~

On hcc/AMD platforms, set the HIP_PRINT_ENV environment variable to 1 and run an application that calls a HIP API to see all HIP-supported environment variables and their current values:

   - HIP_PRINT_ENV = 1: print HIP environment variables
   - HIP_TRACE_API = 1: trace each HIP API call. Print the function name and return code to stderr as the program executes.
   - HIP_LAUNCH_BLOCKING = 0: make HIP APIs Â“host-synchronousÂ” so they are blocked until any kernel launches or data-copy commands are complete (an alias is CUDA_LAUNCH_BLOCKING)

   - KMDUMPISA = 1 : Will dump the GCN ISA for all kernels into the local directory. (This flag is provided by HCC).


Debugging hipcc
~~~~~~~~~~~~~~~~~~~~

To see the detailed commands that hipcc issues, set the environment variable HIPCC_VERBOSE to 1. Doing so will print to stderr the hcc (or nvcc) commands that hipcc generates. 

::

    export HIPCC_VERBOSE=1 make¦ hipcc-cmd: /opt/hcc/bin/hcc -hc
    -I/opt/hcc/include -stdlib=libc++ -I../../../../hc/include
    -I../../../../include/hcc_detail/cuda -I../../../../include -x c++
    -I../../common -O3 -c backprop_cuda.cu



What Does This Error Mean?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

/usr/include/c++/v1/memory:5172:15: error: call to implicitly deleted default constructor of 'std::__1::bad_weak_ptr' throw bad_weak_ptr();

If you pass a ".cu" file, hcc will attempt to compile it as a CUDA language file. You must tell hcc that it's in fact a C++ file: use the "-x c++" option.


HIP Environment Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On the HCC path, HIP provides a number of environment variables that control
the behavior of HIP.  Some of these are useful for application development
(for example HIP_VISIBLE_DEVICES, HIP_LAUNCH_BLOCKING), some are useful for
performance tuning or experimentation (for example HIP_STAGING*), and some are
useful for debugging (HIP_DB).  You can see the environment variables
supported by HIP as well as their current values and usage with the
environment var "HIP_PRINT_ENV" - set this and then run any HIP application.  

For example:

$ HIP_PRINT_ENV=1 ./myhipapp

.. todo clean up

HIP_PRINT_ENV = 1 : Print HIP environment variables. 
HIP_LAUNCH_BLOCKING = 0 : Make HIP APIs host-synchronous,
so they block until any kernel launches or data copy commands complete.
Alias: CUDA_LAUNCH_BLOCKING.HIP_DB = 0 : Print various debug info.
Bitmask, see hip_hcc.cpp for more information. HIP_TRACE_API = 0 : Trace
each HIP API call. Print function name and return code to stderr as
program executes. HIP_TRACE_API_COLOR = green : Color to use for
HIP_API. None/Red/Green/Yellow/Blue/Magenta/Cyan/White HIP_PROFILE_API =
0 : Add HIP function begin/end to ATP file generated with CodeXL
HIP_VISIBLE_DEVICES = 0 : Only devices whose index is present in the
secquence are visible to HIP applications and they are enumerated in the
order of secquence

\``\`

Editor Highlighting
~~~~~~~~~~~~~~~~~~~
See the utils/vim or utils/gedit directories to add handy highlighting to hip files.


