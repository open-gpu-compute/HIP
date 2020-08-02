Porting a CUDA project to HIP
=============================

In addition to providing a portable C++ programming environment for
GPUs, HIP is designed to ease the porting of existing CUDA code into the
HIP environment. This section describes the available tools and provides
practical suggestions on how to port CUDA code and work through common
issues.

General tips
------------

Starting the port on a CUDA machine is often the easiest approach,
since you can incrementally port pieces of the code to HIP while leaving
the rest in CUDA. (Recall that on CUDA machines HIP is just a thin layer
over CUDA, so the two code types can interoperate on nvcc platforms.)
Also, the HIP port can be compared with the original CUDA code
for function and performance.

Once the CUDA code is ported to HIP and is running on the CUDA machine,
compile the HIP code using the HIP compiler on an AMD machine.

HIP ports can replace CUDA versions: HIP can deliver the same performance
as a native CUDA implementation, with the benefit of portability to both 
Nvidia and AMD architectures as well as a path to future C++ standard
support. You can handle platform-specific features through conditional
compilation or by adding them to the open-source HIP infrastructure.

Use `bin/hipconvertinplace-perl.sh <https://github.com/ROCm-Developer-Tools/HIP/blob/master/bin/hipconvertinplace-perl.sh>`__
to hipify all code files in the CUDA source directory.

Auto convert CUDA source
------------------------

Deteermine which files can be converted
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``hipexamine-perl.sh`` tool will scan a source directory to
determine which files contain CUDA code and how much of that code can be
automatically hipified.

::

   > cd examples/rodinia_3.0/cuda/kmeans
   > $HIP_DIR/bin/hipexamine-perl.sh.
   info: hipify ./kmeans.h =====>
   info: hipify ./unistd.h =====>
   info: hipify ./kmeans.c =====>
   info: hipify ./kmeans_cuda_kernel.cu =====>
     info: converted 40 CUDA->HIP refs( dev:0 mem:0 kern:0 builtin:37 math:0 stream:0 event:0 err:0 def:0 tex:3 other:0 ) warn:0 LOC:185
   info: hipify ./getopt.h =====>
   info: hipify ./kmeans_cuda.cu =====>
     info: converted 49 CUDA->HIP refs( dev:3 mem:32 kern:2 builtin:0 math:0 stream:0 event:0 err:0 def:0 tex:12 other:0 ) warn:0 LOC:311
   info: hipify ./rmse.c =====>
   info: hipify ./cluster.c =====>
   info: hipify ./getopt.c =====>
   info: hipify ./kmeans_clustering.c =====>
   info: TOTAL-converted 89 CUDA->HIP refs( dev:3 mem:32 kern:2 builtin:37 math:0 stream:0 event:0 err:0 def:0 tex:15 other:0 ) warn:0 LOC:3607
     kernels (1 total) :   kmeansPoint(1)

``hipexamine-perl`` scans each code file (cpp, c, h, hpp, etc.) found in the
specified directory:

-  Files with no CUDA code (ie ``kmeans.h``) print one line summary just
   listing the source file name.
-  Files with CUDA code print a summary of what was found - for example
   the ``kmeans_cuda_kernel.cu`` file:

::

   info: hipify ./kmeans_cuda_kernel.cu =====>
     info: converted 40 CUDA->HIP refs( dev:0 mem:0 kern:0 builtin:37 math:0 stream:0 event:0 err:0 def:0 tex:3 other:0 ) warn:0 LOC:185


Interesting information in ``kmeans_cuda_kernel.cu`` :

-  How many CUDA calls were converted to HIP (``40``)
-  Breakdown of the CUDA functionality used (``dev:0 mem:0 ...``). This
   file uses many CUDA builtins (``37``) and texture functions (``3``).
-  Warning for code that looks like CUDA API but was not converted (``0``
   in this file).
-  Count Lines-of-Code (LOC) - ``185`` for this file.

``hipexamine-perl`` also presents a summary at the end of the process for
the statistics collected across all files. This has similar format to
the per-file reporting, and also includes a list of all kernels which
have been called. An example from above:

.. code:: shell

   info: TOTAL-converted 89 CUDA->HIP refs( dev:3 mem:32 kern:2 builtin:37 math:0 stream:0 event:0 err:0 def:0 tex:15 other:0 ) warn:0 LOC:3607
     kernels (1 total) :   kmeansPoint(1)

Converting a project in-place
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: shell

   > hipify-perl --inplace

For each input file FILE, this script will: 

- If FILE.prehip file does not exist, copy the original code to a new file
  with extension .prehip. Then, hipify the code file. 
- If "FILE.prehip" file exists, hipify FILE.prehip and save to FILE.

This is useful for testing improvements to the hipify toolset.

The `hipconvertinplace-perl.sh
<https://github.com/ROCm-Developer-Tools/HIP/blob/master/bin/hipconvertinplace-perl.sh>`__
script will perform inplace conversion for all code files in the specified
directory. This can be quite handy when dealing with an existing CUDA code
base since the script preserves the existing directory structure and filenames
and includes work. After converting in-place, you can review the code to add
additional parameters to directory names.

.. code:: shell

   > hipconvertinplace-perl.sh MY_SRC_DIR
