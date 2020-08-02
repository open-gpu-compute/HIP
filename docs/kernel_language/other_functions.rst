Other Functions
===============

Texture Functions
-----------------

Texture functions are not supported.

Surface Functions
-----------------

Surface functions are not supported.

Timer Functions
---------------

HIP provides the following built-in functions for reading a high-resolution
timer from the device. ::

  clock_t clock()
  long long int clock64()

Returns the value of counter that is incremented every clock cycle on
device. Difference in values returned provides the cycles used.

Atomic Functions
----------------

Atomic functions execute as read-modify-write operations residing in global
or shared memory. No other device or thread can observe or modify the memory
location during an atomic operation. If multiple instructions from different
devices or threads target the same memory location, the instructions are
serialized in an undefined order.

HIP supports the following atomic operations.

.. csv-table::
   :header-rows: 1

   Function,Supported in HIP,Supported in CUDA
   "int atomicAdd(int* address, int val)",✓,✓
   "unsigned int atomicAdd(unsigned int* address,unsigned int val)",✓,✓
   "unsigned long long int atomicAdd(unsigned long long int* address,unsigned long long int val)",✓,✓
   "float atomicAdd(float* address, float val)",✓,✓
   "int atomicSub(int* address, int val)",✓,✓
   "unsigned int atomicSub(unsigned int* address,unsigned int val)",✓,✓
   "int atomicExch(int* address, int val)",✓,✓
   "unsigned int atomicExch(unsigned int* address,unsigned int val)",✓,✓
   "unsigned long long int atomicExch(unsigned long long int* address,unsigned long long int val)",✓,✓
   "float atomicExch(float* address, float val)",✓,✓
   "int atomicMin(int* address, int val)",✓,✓
   "unsigned int atomicMin(unsigned int* address,unsigned int val)",✓,✓
   "unsigned long long int atomicMin(unsigned long long int* address,unsigned long long int val)",✓,✓
   "int atomicMax(int* address, int val)",✓,✓
   "unsigned int atomicMax(unsigned int* address,unsigned int val)",✓,✓
   "unsigned long long int atomicMax(unsigned long long int* address,unsigned long long int val)",✓,✓
   unsigned int atomicInc(unsigned int* address),✗,✓
   unsigned int atomicDec(unsigned int* address),✗,✓
   "int atomicCAS(int* address, int compare, int val)",✓,✓
   "unsigned int atomicCAS(unsigned int* address,unsigned int compare,unsigned int val)",✓,✓
   "unsigned long long int atomicCAS(unsigned long long int* address,unsigned long long int compare,unsigned long long int val)",✓,✓
   "int atomicAnd(int* address, int val)",✓,✓
   "unsigned int atomicAnd(unsigned int* address,unsigned int val)",✓,✓
   "unsigned long long int atomicAnd(unsigned long long int* address,unsigned long long int val)",✓,✓
   "int atomicOr(int* address, int val)",✓,✓
   "unsigned int atomicOr(unsigned int* address,unsigned int val)",✓,✓
   "unsigned long long int atomicOr(unsigned long long int* address,unsigned long long int val)",✓,✓
   "int atomicXor(int* address, int val)",✓,✓
   "unsigned int atomicXor(unsigned int* address,unsigned int val)",✓,✓
   "unsigned long long int atomicXor(unsigned long long int* address,unsigned long long int val))",✓,✓

.. note:: Caveats and Features Under-Development:

   HIP enables atomic operations on 32-bit integers. Additionally, it
   supports an atomic float add. AMD hardware, however, implements 	 the
   float add using a CAS loop, so this function may not perform efficiently.


 .. _WarpCross:

Warp Cross Lane Functions
-------------------------

Warp cross-lane functions operate across all lanes in a warp. The hardware
guarantees that all warp lanes will execute in lockstep, so additional
synchronization is unnecessary, and the instructions use no shared memory.

Note that Nvidia and AMD devices have different warp sizes, so portable code
should use the warpSize built-ins to query the warp size. Hipified code from
the Cuda path requires careful review to ensure it doesn’t assume a waveSize
of 32. "Wave-aware" code that assumes a waveSize of 32 will run on a wave-64
machine, but it will utilize only half of the machine resources. In addition
to the warpSize device function, host code can obtain the warpSize from the
device properties

.. code-block:: cpp
    
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceID);
    int w = props.warpSize;  
    // implement portable algorithm based on w (rather than assume 32 or 64)

Warp Vote and Ballot Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   int __all(int predicate)
   int __any(int predicate)
   uint64_t __ballot(int predicate)

Threads in a warp are referred to as lanes and are numbered from 0 to
warpSize -- 1. For these functions, each warp lane contributes 1 -- the bit
value (the predicate), which is efficiently broadcast to all lanes in the
warp. The 32-bit int predicate from each lane reduces to a 1-bit value: 0
(predicate = 0) or 1 (predicate != 0). __any and __all provide a summary
view of the predicates that the other warp lanes contribute:

* __any() returns 1 if any warp lane contributes a nonzero predicate,
  or 0 otherwise
* __all() returns 1 if all other warp lanes contribute nonzero
  predicates, or 0 otherwise

Applications can test whether the target platform supports the any/all
instruction using the hasWarpVote device property or the
HIP_ARCH_HAS_WARP_VOTE compiler define.

__ballot provides a bit mask containing the 1-bit predicate value from each
lane. The nth bit of the result contains the 1 bit contributed by the nth
warp lane. Note that HIP's __ballot function supports a 64-bit return value
(compared with Cuda’s 32 bits). Code ported from Cuda should support the
larger warp sizes that the HIP version of this instruction supports.
Applications can test whether the target platform supports the ballot
instruction using the hasWarpBallot device property or the
HIP_ARCH_HAS_WARP_BALLOT compiler define.


Warp Shuffle Functions
~~~~~~~~~~~~~~~~~~~~~~

Half-float shuffles are not supported. The default width is warpSize---see
:ref:`WarpCross` . Applications should not assume the warpSize is 32 or 64.
 
.. code-block:: cpp

   int   __shfl      (int var,   int srcLane, int width=warpSize);
   float __shfl      (float var, int srcLane, int width=warpSize);
   int   __shfl_up   (int var,   unsigned int delta, int width=warpSize);
   float __shfl_up   (float var, unsigned int delta, int width=warpSize);
   int   __shfl_down (int var,   unsigned int delta, int width=warpSize);
   float __shfl_down (float var, unsigned int delta, int width=warpSize) ;
   int   __shfl_xor  (int var,   int laneMask, int width=warpSize) 
   float __shfl_xor  (float var, int laneMask, int width=warpSize);

Profiler Counter Function
-------------------------

The Cuda __prof_trigger() instruction is not supported.

Assert
------

The assert function is under development.

Printf
------

HIP supports the use of *printf* in the device code. The parameters and
return value for the device-side *printf* follow the POSIX.1 standard, with
the exception that the "%n" specifier is not supported.  No host side
runtime calls by the application are needed to cause the output to appear.
There is no limit on the number of device-side calls to *printf* or the
amount of data that is printed.


Device-Side Dynamic Global Memory Allocation
--------------------------------------------

Device-side dynamic global memory allocation is under development. HIP now
includes a preliminary implementation of malloc and free that can be called
from device functions.

__launch_bounds__
~~~~~~~~~~~~~~~~~

GPU multiprocessors have a fixed pool of resources (primarily registers and
shared memory) which are shared by the actively running warps. Using more
resources can increase IPC of the kernel but reduces the resources available
for other warps and limits the number of warps that can be simultaneously
running. Thus GPUs have a complex relationship between resource usage and
performance.

hip_launch_bounds allows the application to provide usage hints that
influence the resources (primarily registers) used by the generated code.
hip_launch_bounds is a function attribute that must be attached to a global
function:

.. code-block:: cpp

  __global__ void `__launch_bounds__`(MAX_THREADS_PER_BLOCK, MIN_WARPS_PER_EU) MyKernel(...) ...
    MyKernel(hipGridLaunch lp, ...) 
    ...

launch_bounds supports two parameters:

*  **MAX_THREADS_PER_BLOCK** - The programmers guarantees that kernel will be
   launched with threads less than MAX_THREADS_PER_BLOCK. (On NVCC this maps
   to the .maxntid PTX directive). If no launch_bounds is specified,
   MAX_THREADS_PER_BLOCK is the maximum block size supported by the device
   (typically 1024 or larger). Specifying MAX_THREADS_PER_BLOCK less than
   the maximum effectively allows the compiler to use more resources than a
   default unconstrained compilation that supports all possible block sizes
   at launch time. The threads-per-block is the product of (hipBlockDim_x *
   hipBlockDim_y * hipBlockDim_z).
*  **MIN_WARPS_PER_EU ** - directs the compiler
   to minimize resource usage so that the requested number of warps can be
   simultaneously active on a multi-processor. Since active warps compete
   for the same fixed pool of resources, the compiler must reduce resources
   required by each warp(primarily registers). MIN_WARPS_PER_EU is optional
   and defaults to 1 if not specified. Specifying a MIN_WARPS_PER_EU greater
   than the default 1 effectively constrains the compiler's resource usage.

Compiler Impact
~~~~~~~~~~~~~~~

The compiler uses these parameters as follows:

* The compiler uses the hints only to manage register usage, and does not
  automatically reduce shared memory or other resources.
  
* Compilation fails if compiler cannot generate a kernel which meets the
  requirements of the specified launch bounds.
  
* From MAX_THREADS_PER_BLOCK, the compiler derives the maximum number of
  warps/block that can be used at launch time. Values of
  MAX_THREADS_PER_BLOCK less than the default allows the compiler to use a
  larger pool of registers : each warp uses registers, and this hint
  contains the launch to a warps/block size which is less than maximum.
  
* From MIN_WARPS_PER_EU, the compiler derives a maximum number of
  registers that can be used by the kernel (to meet the required
  simultaneous active blocks). If MIN_WARPS_PER_EU is 1, then the kernel
  can use all registers supported by the multiprocessor.
  
* The compiler ensures that the registers used in the kernel is less than
  both allowed maximums, typically by spilling registers 	 (to shared or
  global memory), or by using more instructions.
  
* The compiler may use heuristics to increase register usage, or may
  simply be able to avoid spilling. The MAX_THREADS_PER_BLOCK 	 is
  particularly useful in this cases, since it allows the compiler to use
  more registers and avoid situations where the compiler 	   constrains
  the register usage (potentially spilling) to meet the requirements of a
  large block size that is never used at launch 	   time.

CU and EU Definitions
~~~~~~~~~~~~~~~~~~~~~

A compute unit (CU) is responsible for executing the waves of a work-group.
It is composed of one or more execution units (EU) which are responsible for
executing waves. An EU can have enough resources to maintain the state of
more than one executing wave. This allows an EU to hide latency by switching
between waves in a similar way to symmetric multithreading on a CPU. In
order to allow the state for multiple waves to fit on an EU, the resources
used by a single wave have to be limited. Limiting such resources can allow
greater latency hiding, but can result in having to spill some register
state to memory. This attribute allows an advanced developer to tune the
number of waves that are capable of fitting within the resources of an EU.
It can be used to ensure at least a certain number will fit to help hide
latency, and can also be used to ensure no more than a certain number will
fit to limit cache thrashing.

Porting from CUDA __launch_bounds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CUDA defines a __launch_bounds which is also designed to control occupancy:

.. code-block:: cpp

   __launch_bounds(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_MULTIPROCESSOR)

The second parameter __launch_bounds parameters must be converted to the
format used __hip_launch_bounds, which uses warps and 	 execution-units
rather than blocks and multi-processors ( This conversion is performed
automatically by the clang hipify tools.)

.. code-block:: cpp
   
   MIN_WARPS_PER_EXECUTION_UNIT = (MIN_BLOCKS_PER_MULTIPROCESSOR * MAX_THREADS_PER_BLOCK)/32


The key differences in the interface are:

* Warps (rather than blocks): The developer is trying to tell the compiler
  to control resource utilization to guarantee some     	 amount of active
  Warps/EU for latency hiding. Specifying active warps in terms of blocks
  appears to hide the micro-architectural 	   details of the warp size, but
  makes the interface more confusing since the developer ultimately needs to
  compute the number of 	 warps to obtain the desired level of control.
  
* Execution Units (rather than multiProcessor): The use of execution units
  rather than multiprocessors provides support for 	    	architectures with
  multiple execution units/multi-processor. For example, the AMD GCN
  architecture has 4 execution units per    	multiProcessor. The
  hipDeviceProps has a field executionUnitsPerMultiprocessor.
  Platform-specific coding techniques such as     	#ifdef can be used to
  specify different launch_bounds for NVCC and HCC platforms, if desired.

maxregcount
~~~~~~~~~~~

Unlike nvcc, hcc does not support the "--maxregcount" option. Instead, users
are encouraged to use the hip_launch_bounds directive since the parameters
are more intuitive and portable than micro-architecture details like
registers, and also the directive allows per-kernel control rather than an
entire file. hip_launch_bounds works on both hcc and nvcc targets.


Register Keyword
----------------

The register keyword is deprecated in C++, and is silently ignored by both
nvcc and hcc. To see warnings, you can pass the option -Wdeprecated-register
to hcc.

Pragma Unroll
-------------

Unroll with a bounds that is known at compile-time is supported. For example

.. code-block:: cpp

  #pragma unroll 16 /* hint to compiler to unroll next loop by 16 */
  for (int i=0; i<16; i++) ...

.. code-block:: cpp
  
  #pragma unroll 1  /* tell compiler to never unroll the loop */
  for (int i=0; i<16; i++) ...

Unbounded loop unroll is under development on HCC compiler. 

.. code-block:: cpp
  
  #pragma unroll /* hint to compiler to completely unroll next loop. */
  for (int i=0; i<16; i++) ...

In-Line Assembly
----------------

In-line assembly, including in-line PTX, in-line HSAIL and in-line GCN ISA,
is not supported. Users who need these features should employ conditional
compilation to provide different functionally equivalent implementations on
each target platform.

C++ Support
-----------

The following C++ features are not supported:

  * Run-time-type information (RTTI)
  * Virtual functions
  * Try / catch

Kernel Compilation
------------------

hipcc now supports compiling C++/HIP kernels to binary code objects. The
user can specify the target for which the binary can be generated. HIP/HCC
does not yet support fat binaries so only a single target may be specified.
The file format for binary is .co which means Code Object. The following
command builds the code object using hipcc.

.. code-block:: cpp

   hipcc --genco --target-isa=[TARGET GPU] [INPUT FILE] -o [OUTPUT FILE]

.. code-block:: cpp
   
   [INPUT FILE] = Name of the file containing kernels
   [OUTPUT FILE] = Name of the generated code object file


Note that one important fact to remember when using binary code objects is
that the number of arguments to the kernel are different on HCC and NVCC
path. Refer to the sample in samples/0_Intro/module_api for differences in
the arguments to be passed to the kernel.


