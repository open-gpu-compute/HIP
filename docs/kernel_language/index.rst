Kernel Language
===============

This section describes the built-in variables and functions accessible from
the HIP kernel. It’s intended for readers who are familiar with Cuda kernel
syntax and want to understand how HIP is different.

Features are marked with one of the following keywords:

* Supported---HIP supports the feature with a Cuda-equivalent function
* Not supported---HIP does not support the feature
* Under development---the feature is under development but not yet available

.. toctree::
    :maxdepth: 2
    
    keywords_qualifiers_variables.rst
    math_functions.rst
    other_functions.rst
    CUDA_Driver_API_functions_supported_by_HIP.md
    CUDA_Runtime_API_functions_supported_by_HIP.md