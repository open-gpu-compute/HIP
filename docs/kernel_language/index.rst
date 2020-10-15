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
 
.. _HIP-API:

HIP API Documentation
######################


.. toctree::
    :maxdepth: 2
    :caption: Contents:

    cuda_HIP_API_compare.rst
    HIP_API/Device-management
    HIP_API/Error
    HIP_API/Stream-Management
    HIP_API/Event-Management
    HIP_API/Memory-Management
    HIP_API/Device-Memory-Access
    HIP_API/Initialization-and-Version
    HIP_API/Context-Management
    HIP_API/Control