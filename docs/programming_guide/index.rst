Programming Guide
=================

Heterogeneous-Computing Interface for Portability aka HIP is a C++
dialect designed to ease conversion of Cuda applications to portable C++ code.
It provides a C-style API and a C++ kernel language. The C++ interface can use
templates and classes across the host/kernel boundary.

The HIPify tool automates much of the conversion work by performing a
source-to-source transformation from Cuda to HIP. HIP code can run on AMD
hardware (through the HCC compiler) or Nvidia hardware (through the NVCC
compiler) with no performance loss compared with the original Cuda code.

Programmers familiar with other GPGPU languages will find HIP very easy to
learn and use. AMD platforms implement this language using the HC dialect
described above, providing similar low-level control over the machine.


**When to use HIP**

Use HIP when converting Cuda applications to portable C++ and for new projects
that require portability between AMD and NVIDIA. HIP provides a C++
development language and access to the best development tools on both
platforms.


**How to install**

..todo::
    write more comprehensive installation instructions


For HIP installation instructions, refer the AMD ROCm Installation Guide at
https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html#hip-installation-instructions

.. toctree::
    :maxdepth: 2

    porting_from_cuda_to_hip.rst
    compiler_directives.rst
    compiler_runtime.rst
    compiler_linking.rst
    memory_management.rst
    workarounds_and_tips.rst
    known_bugs.rst
    debug.rst
    profiling.rst