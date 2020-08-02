.. HIP documentation master file, created by
   sphinx-quickstart on Sat Aug  1 10:27:00 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to HIP's documentation!
===============================

.. toctree::
   :maxdepth: 2
   :hidden:

   programming_guides/index.rst
   programming_guides/faq.rst
   programming_guides/comparison.rst
   programming_guides/porting_from_cuda_to_hip.rst
   programming_guides/compiler_directives.rst
   programming_guides/compiler_runtime.rst
   programming_guides/compiler_linking.rst
   programming_guides/memory_management.rst
   programming_guides/workarounds_and_tips.rst
   programming_guides/repository_information.rst
   programming_guides/api_overview/index.rst


HIP is a C++ Runtime API and Kernel Language that allows developers
to create portable applications for AMD and NVIDIA GPUs from single
source code.

 * HIP is very thin and has little or no performance impact over coding
   directly in CUDA or hcc "HC" mode.
 * HIP allows coding in a single-source C++ programming language including
   features such as templates, C++11 lambdas, classes, namespaces, and more.
 * HIP allows developers to use the "best" development environment and tools
   on each target platform.
 * The "hipify" tool automatically converts source from CUDA to HIP.
 * Developers can specialize for the platform (CUDA or hcc) to tune for
   performance or handle tricky cases

New projects can be developed directly in the portable HIP C++ language and
can run on either NVIDIA or AMD platforms. Additionally, HIP provides porting
tools which make it easy to port existing CUDA codes to the HIP layer, with no
loss of performance as compared to the original CUDA application. HIP is not
intended to be a drop-in replacement for CUDA, and developers should expect to
do some manual coding and performance tuning work to complete the port.

HIP provides a C++ syntax that is suitable for compiling most code that
commonly appears in compute kernels, including classes, namespaces, operator
overloading, templates and more. Additionally, it defines other language
features designed specifically to target accelerators, such as the following:

* A kernel-launch syntax that uses standard C++, resembles a function call
  and is portable to all HIP targets
* Short-vector headers that can serve on a host or a device
* Math functions resembling those in the "math.h" header included with
  standard C++ compilers
* Built-in functions for accessing specific GPU hardware capabilities