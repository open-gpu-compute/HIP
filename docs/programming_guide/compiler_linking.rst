Compiler linking
================

Linking Issues
--------------

Linking With hipcc
~~~~~~~~~~~~~~~~~~

hipcc adds the necessary libraries for HIP as well as for the
accelerator compiler (nvcc or AMD compiler). We recommend linking with
hipcc since it automatically links the binary to the necessary HIP
runtime libraries. It also has knowledge on how to link and to manage
the GPU objects.

-lm Option
~~~~~~~~~~

hipcc adds -lm by default to the link command.

Linking Code With Other Compilers
---------------------------------

CUDA code often uses nvcc for accelerator code (defining and launching
kernels, typically defined in .cu or .cuh files). It also uses a
standard compiler (g++) for the rest of the application. nvcc is a
preprocessor that employs a standard host compiler (gcc) to generate the
host code. Code compiled using this tool can employ only the
intersection of language features supported by both nvcc and the host
compiler. In some cases, you must take care to ensure the data types and
alignment of the host compiler are identical to those of the device
compiler. Only some host compilers are supported for example, recent
nvcc versions lack Clang host-compiler capability.

hcc generates both device and host code using the same Clang-based
compiler. The code uses the same API as gcc, which allows code generated
by different gcc-compatible compilers to be linked together. For
example, code compiled using hcc can link with code compiled using
standard compilers (such as gcc, ICC and Clang). Take care to ensure
all compilers use the same standard C++ header and library formats.

libc++ and libstdc++
~~~~~~~~~~~~~~~~~~~~

hipcc links to libstdc++ by default. This provides better compatibility
between g++ and HIP.

If you pass “stdlib=libc++" to hipcc, hipcc will use the libc++
library. Generally, libc++ provides a broader set of C++ features while
libstdc++ is the standard for more compilers (notably including g++).

When cross-linking C++ code, any C++ functions that use types from the
C++ standard library (including std::string, std::vector and other
containers) must use the same standard-library implementation. They
include the following:

-  Functions or kernels defined in hcc that are called from a standard
   compiler
-  Functions defined in a standard compiler that are called from hcc.

Applications with these interfaces should use the default libstdc++
linking.

Applications which are compiled entirely with hipcc, and which benefit
from advanced C++ features not supported in libstdc++, and which do not
require portability to nvcc, may choose to use libc++.

HIP Headers (hip_runtime.h, hip_runtime_api.h)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The hip_runtime.h and hip_runtime_api.h files define the types,
functions and enumerations needed to compile a HIP program:

-  hip_runtime_api.h: defines all the HIP runtime APIs (e.g., hipMalloc)
   and the types required to call them. A source file that is only
   calling HIP APIs but neither defines nor launches any kernels can
   include hip_runtime_api.h. hip_runtime_api.h uses no custom hc
   language features and can be compiled using a standard C++ compiler.
   
-  hip_runtime.h: included in hip_runtime_api.h. It additionally
   provides the types and defines required to create and launch kernels.
   hip_runtime.h does use custom hc language features, but they are
   guarded by ifdef checks. It can be compiled using a standard C++
   compiler but will expose a subset of the available functions.

CUDA has slightly different contents for these two files. In some cases
you may need to convert hipified code to include the richer
hip_runtime.h instead of hip_runtime_api.h.

Using a Standard C++ Compiler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can compile hip_runtime_api.h using a standard C or C++ compiler
(e.g., gcc or ICC). The HIP include paths and defines
(``__HIP_PLATFORM_HCC__`` or ``__HIP_PLATFORM_NVCC__``) must pass to the
standard compiler; hipconfig then returns the necessary options:

::

   > hipconfig --cxx_config
    -D__HIP_PLATFORM_HCC__ -I/home/user1/hip/include

You can capture the hipconfig output and passed it to the standard
compiler; below is a sample makefile syntax:

::

   CPPFLAGS += $(shell $(HIP_PATH)/bin/hipconfig --cpp_config)

nvcc includes some headers by default. However, HIP does not include
default headers, and instead all required files must be explicitly
included. Specifically, files that call HIP run-time APIs or define HIP
kernels must explicitly include the appropriate HIP headers. If the
compilation process reports that it cannot find necessary APIs (for
example, error: identifier ˜hipSetDevice™ is undefined), ensure that
the file includes hip_runtime.h (or hip_runtime_api.h, if appropriate).
The hipify-perl script automatically converts cuda_runtime.h to
hip_runtime.h, and it converts â€œcuda_runtime_api.h to
hip_runtime_api.h, but it may miss nested headers or macros.

cuda.h
^^^^^^

The hcc path provides an empty cuda.h file. Some existing CUDA programs
include this file but does not require any of the functions.

Choosing HIP File Extensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Many existing CUDA projects use the .cu and .cuh file extensions to
indicate code that should be run through the nvcc compiler. For quick
HIP ports, leaving these file extensions unchanged is often easier, as
it minimizes the work required to change file names in the directory and
#include statements in the files.

For new projects or ports which can be re-factored, we recommend the use
of the extension .hip.cpp for source files, and .hip.h or .hip.hpp
for header files. This indicates that the code is standard C++ code, but
also provides a unique indication for make tools to run hipcc when
appropriate.
