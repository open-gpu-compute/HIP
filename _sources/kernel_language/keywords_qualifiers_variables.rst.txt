Keywords, Qualifiers and Variables
==================================

Function-Type Qualifiers 
------------------------
   
__device__
~~~~~~~~~~

Supported __device__ functions are

* Executed on the device
* Called from the device only

The __device__ keyword can combine with the host keyword.

__global__
~~~~~~~~~~

Supported __global__ functions are

* Executed on the device
* Called ("launched") from the host

HIP __global__ functions must have a void return type.

HIP lacks dynamic-parallelism support, so __global__ functions cannot be
called from the device.

__host__
~~~~~~~~

Supported __host__ functions are

* Executed on the host
* Called from the host

__host__ can combine with __device__, in which case the function compiles
for both the host and device. These functions cannot use the HIP grid
coordinate functions (for example, ``hipThreadIdx_x``). A possible workaround
is to pass the necessary coordinate info as an argument to the function.

__host__ cannot combine with __global__.

HIP parses the __noinline__ and __forceinline__ keywords and converts them
to the appropriate Clang attributes. The hcc compiler, however, currently
in-lines all device functions, so they are effectively ignored.

Calling __global__ Functions
----------------------------

__global__ functions are often referred to as kernels, and calling one is
termed launching the kernel. These functions require the caller to specify
an "execution configuration" that includes the grid and block dimensions.
The execution configuration can also include other information for the
launch, such as the amount of additional shared memory to allocate and the
stream where the kernel should execute. HIP introduces a standard C++
calling convention to pass the execution configuration to the kernel (this
convention replaces the Cuda <<< >>> syntax).

In HIP, Kernels launch with the "hipLaunchKernelGGL" function The first five
parameters to hipLaunchKernelGGL are the following:

*  **symbol kernelName**: the name of the kernel to launch. To support
   template kernels which contains ``,`` use the HIP_KERNEL_NAME
   macro. The hipify tools insert this automatically.
*  **dim3 gridDim**: 3D-grid dimensions specifying the number of
   blocks to launch.
*  **dim3 blockDim**: 3D-block dimensions specifying the number of threads 
   in each block.
*  **size_t dynamicShared**: amount of additional shared memory to allocate
   when launching the kernel
*  **hipStream_t**: stream where the kernel should execute. A value of 0
   corresponds to the NULL stream(see :ref:`Synchronization-Functions`).

Kernel arguments follow these first five parameters

.. code-block:: cpp
    
      //Example pseudo code introducing hipLaunchKernelGGL
      __global__ MyKernel(float *A, float *B, float *C, size_t N)
      {
      ...
      } 
      //Replace MyKernel<<<dim3(gridDim), dim3(gridDim), 0, 0>>> (a,b,c,n);
      hipLaunchKernelGGL(MyKernel, dim3(gridDim), dim3(groupDim), 0/*dynamicShared*/, 0/*stream), a, b, c, n)


The hipLaunchKernelGGL macro always starts with the five parameters
specified above, followed by the kernel arguments. The Hipify script
automatically converts Cuda launch syntax to hipLaunchKernelGGL, including
conversion of optional arguments in <<< >>> to the five required
hipLaunchKernelGGL parameters. The :ref:`dim3` constructor accepts zero to
three arguments and will by default initialize unspecified dimensions to 1.
See dim3. The kernel uses the coordinate built-ins (hipThread*, hipBlock*,
hipGrid*) to determine coordinate index and coordinate bounds of the work
item that’s currently executing. 

 .. _Kernel:

Kernel-Launch Example
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

    // Example showing use of host/device function
    __host__ __device__
    float PlusOne(float x)
    {
       return x + 1.0;
    }

    __global__
    void
    MyKernel (const float *a, const float *b, float *c, unsigned N)
    {
       unsigned gid = hipThreadIdx_x; // <- coordinate index function
       if (gid < N) {
           c[gid] = a[gid] + PlusOne(b[gid]);
       }
    }
    void callMyKernel()
    {
        float *a, *b, *c; // initialization not shown...
        unsigned N = 1000000;
        const unsigned blockSize = 256;
        hipLaunchKernelGGL(MyKernel,
        (N/blockSize), dim3(blockSize), 0, 0,  a,b,c,N);
    }


 

Variable-Type Qualifiers
------------------------

__constant__
~~~~~~~~~~~~
 
The __constant__ keyword is supported. The host writes constant memory
before launching the kernel; from the GPU, this memory is read-only during
kernel execution. The functions for accessing constant memory
(hipGetSymbolAddress(), hipGetSymbolSize(), hipMemcpyToSymbol(),
hipMemcpyToSymbolAsync, hipMemcpyFromSymbol, hipMemcpyFromSymbolAsync) are
under development.

__shared__
~~~~~~~~~~

The __shared__ keyword is supported. extern __shared__ allows the host to
dynamically allocate shared memory and is specified as a launch parameter.
HIP uses an alternate syntax based on the HIP_DYNAMIC_SHARED macro.

__managed__
~~~~~~~~~~~

Managed memory, including the __managed__ keyword, are not supported in HIP.

__restrict__
~~~~~~~~~~~~

The __restrict__ keyword tells the compiler that the associated memory
pointer will not alias with any other pointer in the kernel or function.
This feature can help the compiler generate better code. In most cases, all
pointer arguments must use this keyword to realize the benefit. hcc support
for the __restrict__ qualifier on kernel arguments is under development.

Built-In Variables
------------------

Coordinate Built-Ins
~~~~~~~~~~~~~~~~~~~~

These built-ins determine the coordinate of the active work item in the
execution grid. They are defined in hip_runtime.h (rather than being
implicitly defined by the compiler).

=============== ==============
 HIP Syntax      Cuda Syntax
===============	==============
hipThreadIdx_x 	 threadIdx.x
hipThreadIdx_y 	 threadIdx.y
hipThreadIdx_z 	 threadIdx.z
	
hipBlockIdx_x 	 blockIdx.x

hipBlockIdx_y 	 blockIdx.y

hipBlockIdx_z 	 blockIdx.z
	
hipBlockDim_x 	 blockDim.x

hipBlockDim_y 	 blockDim.y

hipBlockDim_z 	 blockDim.z
	
hipGridDim_x 	 gridDim.x

hipGridDim_y 	 gridDim.y

hipGridDim_z 	 gridDim.z
=============== ==============

warpSize
~~~~~~~~

The warpSize variable is of type int and contains the warp size (in threads)
for the target device. Note that all current Nvidia devices return 32 for
this variable, and all current AMD devices return 64. Device code should use
the warpSize built-in to develop portable wave-aware code.

Vector Types
------------

Note that these types are defined in hip_runtime.h and are not automatically
provided by the compiler.

Short Vector Types
~~~~~~~~~~~~~~~~~~

Short vector types derive from the basic integer and floating-point types.
They are structures defined in hip_vector_types.h. The first, second, third
and fourth components of the vector are accessible through the x, y, z and w
fields, respectively. All the short vector types support a constructor
function of the form make_<type_name>(). For example, float4
make_float4(float x, float y, float z, float w) creates a vector of type
float4 and value (x,y,z,w). HIP supports the following short vector formats:

* Signed Integers:
    * char1, char2, char3, char4
    * short1, short2, short3, short4
    * int1, int2, int3, int4
    * long1, long2, long3, long4
    * longlong1, longlong2, longlong3, longlong4
* Unsigned Integers:
    * uchar1, uchar2, uchar3, uchar4
    * ushort1, ushort2, ushort3, ushort4
    * uint1, uint2, uint3, uint4
    * ulong1, ulong2, ulong3, ulong4
    * ulonglong1, ulonglong2, ulonglong3, ulonglong4
* Floating Points
    * float1, float2, float3, float4
    * double1, double2, double3, double4

 .. _dim3:

dim3
~~~~ 

dim3 is a three-dimensional integer vector type commonly used to
specify grid and group dimensions. Unspecified dimensions are initialized to
1.

.. code-block:: cpp

 typedef struct dim3 {
   uint32_t x; 
   uint32_t y; 
   uint32_t z; 

   dim3(uint32_t _x=1, uint32_t _y=1, uint32_t _z=1) : x(_x), y(_y), z(_z) {};
 };


Memory-Fence Instructions
-------------------------

HIP supports __threadfence() and __threadfence_block().

HIP provides workaround for threadfence_system() under HCC path. To enable
the workaround, HIP should be built with environment variable
HIP_COHERENT_HOST_ALLOC enabled. In addition,the kernels that use
__threadfence_system() should be modified as follows:

* The kernel should only operate on finegrained system memory; which
  should be allocated with hipHostMalloc().
* Remove all memcpy for those allocated finegrained system memory regions.

 .. _Synchronization-Functions:

Synchronization Functions
-------------------------

The __syncthreads() built-in function is supported in HIP. The
__syncthreads_count(int), __syncthreads_and(int) and __syncthreads_or(int)
functions are under development.