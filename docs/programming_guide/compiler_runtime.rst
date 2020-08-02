Compiler Runtime
================

Finding HIP
-----------

Makefiles can use the following syntax to conditionally provide a default
HIP_PATH if one does not exist:

::

   HIP_PATH ?= $(shell hipconfig --path)

Identifying HIP Runtime
-----------------------

HIP can depend on ROCclr, or NVCC as runtime

-  **AMD platform**: ``HIP_ROCclr`` is defined on AMD platform that HIP
   use Radeon Open Compute Common Language Runtime, called ROCclr.
   ROCclr is a virtual device interface that HIP runtimes interact with
   different backends which allows runtimes to work on Linux , as well
   as Windows without much efforts.

-  **NVIDIA platform**: On Nvidia platform, HIP is just a thin layer on top
   of CUDA. On non-AMD platform, HIP runtime determines if nvcc is
   available and can be used. If available, HIP_PLATFORM is set to
   nvcc and underneath CUDA path is used.

hipLaunchKernel
---------------

hipLaunchKernel is a variadic macro which accepts as parameters the launch
configurations (grid dims, group dims, stream, dynamic shared size) followed
by a variable number of kernel arguments. This sequence is then expanded
into the appropriate kernel launch syntax depending on the platform. While
this can be a convenient single-line kernel launch syntax, the macro
implementation can cause issues when nested inside other macros. For
example, consider the following:

.. code:: cpp

    // Will cause compile error:
    #define MY_LAUNCH(command, doTrace) \
    {\
        if (doTrace) printf ("TRACE: %s\n", #command); \
        (command);   /* The nested ( ) will cause compile error */\
    }

   MY_LAUNCH (hipLaunchKernel(vAdd, dim3(1024), dim3(1), 0, 0, Ad), true, "firstCall");

Avoid nesting macro parameters inside parenthesis - here's an
alternative that will work:

.. code:: cpp

   #define MY_LAUNCH(command, doTrace) \
   {\
       if (doTrace) printf ("TRACE: %s\n", #command); \
       command;\ 
   }

   MY_LAUNCH (hipLaunchKernel(vAdd, dim3(1024), dim3(1), 0, 0, Ad), true, "firstCall");

Compiler Options
----------------

hipcc is a portable compiler driver that will call nvcc or HIP-Clang
(depending on the target system) and attach all required include and
library options. It passes options through to the target compiler. Tools
that call hipcc must ensure the compiler options are appropriate for the
target compiler. The ``hipconfig`` script may helpful in identifying the
target platform, compiler and runtime. It can also help set options
appropriately.
