Repository Information
======================

The HIP repository maintains several branches. The branches that are of importance are:

 * master branch: This is the stable branch. All stable releases are based on
   this branch.
 * developer-preview branch: This is the branch were the new
   features still under development are visible. While this maybe of interest to
   many, it should be noted that this branch and the features under development
   might not be stable.

Release tagging
---------------
HIP releases are typically of two types. The tag naming convention is different for both types of releases to help differentiate them.

* release_x.yy.zzzz: These are the stable releases based on the master branch. This type of release is typically made once a month.
* preview_x.yy.zzzz: These denote pre-release code and are based on the developer-preview branch. This type of release is typically made once a week.

HCC deprecetation notice
------------------------

.. warning::
    In the v3.5 release, the Heterogeneous Compute Compiler (HCC) compiler is
    deprecated and the HIP-Clang compiler is introduced for compiling
    Heterogeneous-Compute Interface for Portability (HIP) programs.

    The HCC environment variables will be gradually deprecated in
    subsequent releases.


The majority of the codebase for the HIP-Clang compiler has been upstreamed to
the Clang trunk. The HIP-Clang implementation has undergone a strict code
review by the LLVM/Clang community and comprehensive tests consisting of
LLVM/Clang build bots. These reviews and tests resulted in higher
productivity, code quality, and lower cost of maintenance.

`FAQ Transition to HIP from HCC <https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-FAQ.html#hip-faq>`_

`HIP Porting Guide <https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-porting-guide.html#hip-porting-guide>`_

