Math Functions
--------------

hcc supports a set of math operations callable from the device.

Single precision mathematical functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Following is the list of supported single precision mathematical functions.

.. csv-table::
   :header-rows: 1

   Function,Supported on Host,Supported on Device
   "float acosf ( float x )

   Calculate the arc cosine of the input argument.",✓,✓
   "float acoshf ( float x )

   Calculate the nonnegative arc hyperbolic cosine of the input argument.",✓,✓
   "float asinf ( float x )

   Calculate the arc sine of the input argument.",✓,✓
   "float asinhf ( float x )

   Calculate the arc hyperbolic sine of the input argument.",✓,✓
   "float atan2f ( float y, float x )

   Calculate the arc tangent of the ratio of first and second input arguments.",✓,✓
   "float atanf ( float x )

   Calculate the arc tangent of the input argument.",✓,✓
   "float atanhf ( float x )

   Calculate the arc hyperbolic tangent of the input argument.",✓,✓
   "float cbrtf ( float x )

   Calculate the cube root of the input argument.",✓,✓
   "float ceilf ( float x )

   Calculate ceiling of the input argument.",✓,✓
   "float copysignf ( float x, float y )

   Create value with given magnitude, copying sign of second value.",✓,✓
   "float cosf ( float x )

   Calculate the cosine of the input argument.",✓,✓
   "float coshf ( float x )

   Calculate the hyperbolic cosine of the input argument.",✓,✓
   "float erfcf ( float x )

   Calculate the complementary error function of the input argument.",✓,✓
   "float erff ( float x )

   Calculate the error function of the input argument.",✓,✓
   "float exp10f ( float x )

   Calculate the base 10 exponential of the input argument.",✓,✓
   "float exp2f ( float x )

   Calculate the base 2 exponential of the input argument.",✓,✓
   "float expf ( float x )

   Calculate the base e exponential of the input argument.",✓,✓
   "float expm1f ( float x )

   Calculate the base e exponential of the input argument, minus 1.",✓,✓
   "float fabsf ( float x )

   Calculate the absolute value of its argument.",✓,✓
   "float fdimf ( float x, float y )

   Compute the positive difference between x and y.",✓,✓
   "float floorf ( float x )

   Calculate the largest integer less than or equal to x.",✓,✓
   "float fmaf ( float x, float y, float z )

   Compute x × y + z as a single operation.",✓,✓
   "float fmaxf ( float x, float y )

   Determine the maximum numeric value of the arguments.",✓,✓
   "float fminf ( float x, float y )

   Determine the minimum numeric value of the arguments.",✓,✓
   "float fmodf ( float x, float y )

   Calculate the floating-point remainder of x / y.",✓,✓
   "float frexpf ( float x, int* nptr )

   Extract mantissa and exponent of a floating-point value.",✓,✗
   "float hypotf ( float x, float y )

   Calculate the square root of the sum of squares of two arguments.",✓,✓
   "int ilogbf ( float x )
   Compute the unbiased integer exponent of the argument.",✓,✓
   "__RETURN_TYPE1 isfinite ( float a )
   Determine whether argument is finite.",✓,✓
   "__RETURN_TYPE1 isinf ( float a )
   Determine whether argument is infinite.",✓,✓
   "__RETURN_TYPE1 isnan ( float a )
   Determine whether argument is a NaN.",✓,✓
   "float ldexpf ( float x, int exp )

   Calculate the value of x ⋅ 2exp.",✓,✓
   "float log10f ( float x )

   Calculate the base 10 logarithm of the input argument.",✓,✓
   "float log1pf ( float x )

   Calculate the value of loge( 1 + x ).",✓,✓
   "float logbf ( float x )

   Calculate the floating point representation of the exponent of the input argument.",✓,✓
   "float log2f ( float x )

   Calculate the base 2 logarithm of the input argument.",✓,✓
   "float logf ( float x )

   Calculate the natural logarithm of the input argument.",✓,✓
   "float modff ( float x, float* iptr )

   Break down the input argument into fractional and integral parts.",✓,✗
   "float nanf ( const char* tagp )

   Returns “Not a Number”” value.”",✗,✓
   "float nearbyintf ( float x )

   Round the input argument to the nearest integer.",✓,✓
   "float powf ( float x, float y )

   Calculate the value of first argument to the power of second argument.",✓,✓
   "float remainderf ( float x, float y )

   Compute single-precision floating-point remainder.",✓,✓
   "float remquof ( float x, float y, int* quo )

   Compute single-precision floating-point remainder and part of quotient.",✓,✗
   "float roundf ( float x )

   Round to nearest integer value in floating-point.",✓,✓
   "float scalbnf ( float x, int n )

   Scale floating-point input by integer power of two.",✓,✓
   "__RETURN_TYPE1 signbit ( float a )

   Return the sign bit of the input.",✓,✓
   "void sincosf ( float x, float* sptr, float* cptr )

   Calculate the sine and cosine of the first input argument.",✓,✗
   "float sinf ( float x )

   Calculate the sine of the input argument.",✓,✓
   "float sinhf ( float x )

   Calculate the hyperbolic sine of the input argument.",✓,✓
   "float sqrtf ( float x )

   Calculate the square root of the input argument.",✓,✓
   "float tanf ( float x )

   Calculate the tangent of the input argument.",✓,✓
   "float tanhf ( float x )

   Calculate the hyperbolic tangent of the input argument.",✓,✓
   "float truncf ( float x )

   Truncate input argument to the integral part.",✓,✓
   "float tgammaf ( float x )

   Calculate the gamma function of the input argument.",✓,✓
   "float erfcinvf ( float y )

   Calculate the inverse complementary function of the input argument.",✓,✓
   "float erfcxf ( float x )

   Calculate the scaled complementary error function of the input argument.",✓,✓
   "float erfinvf ( float y )

   Calculate the inverse error function of the input argument.",✓,✓
   "float fdividef ( float x, float y )

   Divide two floating point values.",✓,✓
   "float frexpf ( float x, int *nptr )

   Extract mantissa and exponent of a floating-point value.",✓,✓
   "float j0f ( float x )

   Calculate the value of the Bessel function of the first kind of order 0 for the input argument.",✓,✓
   "float j1f ( float x )

   Calculate the value of the Bessel function of the first kind of order 1 for the input argument.",✓,✓
   "float jnf ( int n, float x )

   Calculate the value of the Bessel function of the first kind of order n for the input argument.",✓,✓
   "float lgammaf ( float x )

   Calculate the natural logarithm of the absolute value of the gamma function of the input argument.",✓,✓
   "long long int llrintf ( float x )

   Round input to nearest integer value.",✓,✓
   "long long int llroundf ( float x )

   Round to nearest integer value.",✓,✓
   "long int lrintf ( float x )

   Round input to nearest integer value.",✓,✓
   "long int lroundf ( float x )

   Round to nearest integer value.",✓,✓
   "float modff ( float x, float *iptr )

   Break down the input argument into fractional and integral parts.",✓,✓
   "float nextafterf ( float x, float y )

   Returns next representable single-precision floating-point value after argument.",✓,✓
   "float norm3df ( float a, float b, float c )

   Calculate the square root of the sum of squares of three coordinates of the argument.",✓,✓
   "float norm4df ( float a, float b, float c, float d )

   Calculate the square root of the sum of squares of four coordinates of the argument.",✓,✓
   "float normcdff ( float y )

   Calculate the standard normal cumulative distribution function.",✓,✓
   "float normcdfinvf ( float y )

   Calculate the inverse of the standard normal cumulative distribution function.",✓,✓
   "float normf ( int dim, const float *a )

   Calculate the square root of the sum of squares of any number of coordinates.",✓,✓
   "float rcbrtf ( float x )

   Calculate the reciprocal cube root function.",✓,✓
   "float remquof ( float x, float y, int *quo )

   Compute single-precision floating-point remainder and part of quotient.",✓,✓
   "float rhypotf ( float x, float y )

   Calculate one over the square root of the sum of squares of two arguments.",✓,✓
   "float rintf ( float x )

   Round input to nearest integer value in floating-point.",✓,✓
   "float rnorm3df ( float a, float b, float c )

   Calculate one over the square root of the sum of squares of three coordinates of the argument.",✓,✓
   "float rnorm4df ( float a, float b, float c, float d )

   Calculate one over the square root of the sum of squares of four coordinates of the argument.",✓,✓
   "float rnormf ( int dim, const float *a )

   Calculate the reciprocal of square root of the sum of squares of any number of coordinates.",✓,✓
   "float scalblnf ( float x, long int n )

   Scale floating-point input by integer power of two.",✓,✓
   "void sincosf ( float x, float *sptr, float *cptr )

   Calculate the sine and cosine of the first input argument.",✓,✓
   "void sincospif ( float x, float *sptr, float *cptr )

   Calculate the sine and cosine of the first input argument multiplied by PI.",✓,✓
   "float y0f ( float x )

   Calculate the value of the Bessel function of the second kind of order 0 for the input argument.",✓,✓
   "float y1f ( float x )

   Calculate the value of the Bessel function of the second kind of order 1 for the input argument.",✓,✓
   "float ynf ( int n, float x )

   Calculate the value of the Bessel function of the second kind of order n for the input argument.",✓,✓

[1] __RETURN_TYPE is dependent on compiler. It is usually ‘int’ for C
compilers and ‘bool’ for C++ compilers.

Double precision mathematical functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Following is the list of supported double precision mathematical functions.


.. csv-table::
   :header-rows: 1

   Function,Supported on Host,Supported on Device
   "double acos ( double x )

   Calculate the arc cosine of the input argument.",✓,✓
   "double acosh ( double x )

   Calculate the nonnegative arc hyperbolic cosine of the input argument.",✓,✓
   "double asin ( double x )

   Calculate the arc sine of the input argument.",✓,✓
   "double asinh ( double x )

   Calculate the arc hyperbolic sine of the input argument.",✓,✓
   "double atan ( double x )

   Calculate the arc tangent of the input argument.",✓,✓
   "double atan2 ( double y, double x )

   Calculate the arc tangent of the ratio of first and second input arguments.",✓,✓
   "double atanh ( double x )

   Calculate the arc hyperbolic tangent of the input argument.",✓,✓
   "double cbrt ( double x )

   Calculate the cube root of the input argument.",✓,✓
   "double ceil ( double x )

   Calculate ceiling of the input argument.",✓,✓
   "double copysign ( double x, double y )

   Create value with given magnitude, copying sign of second value.",✓,✓
   "double cos ( double x )

   Calculate the cosine of the input argument.",✓,✓
   "double cosh ( double x )

   Calculate the hyperbolic cosine of the input argument.",✓,✓
   "double erf ( double x )

   Calculate the error function of the input argument.",✓,✓
   "double erfc ( double x )

   Calculate the complementary error function of the input argument.",✓,✓
   "double exp ( double x )

   Calculate the base e exponential of the input argument.",✓,✓
   "double exp10 ( double x )

   Calculate the base 10 exponential of the input argument.",✓,✓
   "double exp2 ( double x )

   Calculate the base 2 exponential of the input argument.",✓,✓
   "double expm1 ( double x )

   Calculate the base e exponential of the input argument, minus 1.",✓,✓
   "double fabs ( double x )

   Calculate the absolute value of the input argument.",✓,✓
   "double fdim ( double x, double y )

   Compute the positive difference between x and y.",✓,✓
   "double floor ( double x )

   Calculate the largest integer less than or equal to x.",✓,✓
   "double fma ( double x, double y, double z )

   Compute x × y + z as a single operation.",✓,✓
   "double fmax ( double , double )

   Determine the maximum numeric value of the arguments.",✓,✓
   "double fmin ( double x, double y )

   Determine the minimum numeric value of the arguments.",✓,✓
   "double fmod ( double x, double y )

   Calculate the floating-point remainder of x / y.",✓,✓
   "double frexp ( double x, int* nptr )

   Extract mantissa and exponent of a floating-point value.",✓,✗
   "double hypot ( double x, double y )

   Calculate the square root of the sum of squares of two arguments.",✓,✓
   "int ilogb ( double x )

   Compute the unbiased integer exponent of the argument.",✓,✓
   "__RETURN_TYPE1 isfinite ( double a )

   Determine whether argument is finite.",✓,✓
   "__RETURN_TYPE1 isinf ( double a )

   Determine whether argument is infinite.",✓,✓
   "__RETURN_TYPE1 isnan ( double a )

   Determine whether argument is a NaN.",✓,✓
   "double ldexp ( double x, int exp )

   Calculate the value of x ⋅ 2exp.",✓,✓
   "double log ( double x )

   Calculate the base e logarithm of the input argument.",✓,✓
   "double log10 ( double x )

   Calculate the base 10 logarithm of the input argument.",✓,✓
   "double log1p ( double x )

   Calculate the value of loge( 1 + x ).",✓,✓
   "double log2 ( double x )

   Calculate the base 2 logarithm of the input argument.",✓,✓
   "double logb ( double x )

   Calculate the floating point representation of the exponent of the input argument.",✓,✓
   "double modf ( double x, double* iptr )

   Break down the input argument into fractional and integral parts.",✓,✗
   "double nan ( const char* tagp )

   Returns “Not a Number”” value.”",✗,✓
   "double nearbyint ( double x )

   Round the input argument to the nearest integer.",✓,✓
   "double pow ( double x, double y )

   Calculate the value of first argument to the power of second argument.",✓,✓
   "double remainder ( double x, double y )

   
   Compute double-precision floating-point remainder.",✓,✓
   "double remquo ( double x, double y, int* quo )

   Compute double-precision floating-point remainder and part of quotient.",✓,✗
   "double round ( double x )

   Round to nearest integer value in floating-point.",✓,✓
   "double scalbn ( double x, int n )

   Scale floating-point input by integer power of two.",✓,✓
   "__RETURN_TYPE1 signbit ( double a )

   Return the sign bit of the input.",✓,✓
   "double sin ( double x )

   Calculate the sine of the input argument.",✓,✓
   "void sincos ( double x, double* sptr, double* cptr )

   Calculate the sine and cosine of the first input argument.",✓,✗
   "double sinh ( double x )

   Calculate the hyperbolic sine of the input argument.",✓,✓
   "double sqrt ( double x )

   Calculate the square root of the input argument.",✓,✓
   "double tan ( double x )

   Calculate the tangent of the input argument.",✓,✓
   "double tanh ( double x )

   Calculate the hyperbolic tangent of the input argument.",✓,✓
   "double tgamma ( double x )

   Calculate the gamma function of the input argument.",✓,✓
   "double trunc ( double x )

   Truncate input argument to the integral part.",✓,✓
   "double erfcinv ( double y )

   Calculate the inverse complementary function of the input argument.",✓,✓
   "double erfcx ( double x )

   Calculate the scaled complementary error function of the input argument.",✓,✓
   "double erfinv ( double y )

   Calculate the inverse error function of the input argument.",✓,✓
   "double frexp ( float x, int *nptr )

   Extract mantissa and exponent of a floating-point value.",✓,✓
   "double j0 ( double x )

   Calculate the value of the Bessel function of the first kind of order 0 for the input argument.",✓,✓
   "double j1 ( double x )

   Calculate the value of the Bessel function of the first kind of order 1 for the input argument.",✓,✓
   "double jn ( int n, double x )

   Calculate the value of the Bessel function of the first kind of order n for the input argument.",✓,✓
   "double lgamma ( double x )

   Calculate the natural logarithm of the absolute value of the gamma function of the input argument.",✓,✓
   "long long int llrint ( double x )

   Round input to nearest integer value.",✓,✓
   "long long int llround ( double x )

   Round to nearest integer value.",✓,✓
   "long int lrint ( double x )

   Round input to nearest integer value.",✓,✓
   "long int lround ( double x )
   
   Round to nearest integer value.",✓,✓
   "double modf ( double x, double *iptr )

   Break down the input argument into fractional and integral parts.",✓,✓
   "double nextafter ( double x, double y )

   Returns next representable single-precision floating-point value after argument.",✓,✓
   "double norm3d ( double a, double b, double c )

   Calculate the square root of the sum of squares of three coordinates of the argument.",✓,✓
   "float norm4d ( double a, double b, double c, double d )

   Calculate the square root of the sum of squares of four coordinates of the argument.",✓,✓
   "double normcdf ( double y )

   Calculate the standard normal cumulative distribution function.",✓,✓
   "double normcdfinv ( double y )

   Calculate the inverse of the standard normal cumulative distribution function.",✓,✓
   "double rcbrt ( double x )

   Calculate the reciprocal cube root function.",✓,✓
   "double remquo ( double x, double y, int *quo )

   Compute single-precision floating-point remainder and part of quotient.",✓,✓
   "double rhypot ( double x, double y )

   Calculate one over the square root of the sum of squares of two arguments.",✓,✓
   "double rint ( double x )

   Round input to nearest integer value in floating-point.",✓,✓
   "double rnorm3d ( double a, double b, double c )

   Calculate one over the square root of the sum of squares of three coordinates of the argument.",✓,✓
   "double rnorm4d ( double a, double b, double c, double d )

   Calculate one over the square root of the sum of squares of four coordinates of the argument.",✓,✓
   "double rnorm ( int dim, const double *a )

   Calculate the reciprocal of square root of the sum of squares of any number of coordinates.",✓,✓
   "double scalbln ( double x, long int n )

   Scale floating-point input by integer power of two.",✓,✓
   "void sincos ( double x, double *sptr, double *cptr )

   Calculate the sine and cosine of the first input argument.",✓,✓
   "void sincospi ( double x, double *sptr, double *cptr )
   
   Calculate the sine and cosine of the first input argument multiplied by PI.",✓,✓
   "double y0f ( double x )

   Calculate the value of the Bessel function of the second kind of order 0 for the input argument.",✓,✓
   "double y1 ( double x )

   Calculate the value of the Bessel function of the second kind of order 1 for the input argument.",✓,✓
   "double yn ( int n, double x )

   Calculate the value of the Bessel function of the second kind of order n for the input argument.",✓,✓


Floating-point intrinsics
~~~~~~~~~~~~~~~~~~~~~~~~~

Following is the list of supported floating-point intrinsics. Note that
intrinsics are supported on device only.

.. csv-table::
   :header-rows: 1

   Function
   "float __cosf ( float x )

   Calculate the fast approximate cosine of the input argument."
   "float __expf ( float x )

   Calculate the fast approximate base e exponential of the input argument."
   "float __frsqrt_rn ( float x )

   Compute 1/√x in round-to-nearest-even mode."
   "float __fsqrt_rd ( float x )

   Compute √x in round-down mode."
   "float __fsqrt_rn ( float x )

   Compute √x in round-to-nearest-even mode."
   "float __fsqrt_ru ( float x )

   Compute √x in round-up mode."
   "float __fsqrt_rz ( float x )

   Compute √x in round-towards-zero mode."
   "float __log10f ( float x )

   Calculate the fast approximate base 10 logarithm of the input argument."
   "float __log2f ( float x )

   Calculate the fast approximate base 2 logarithm of the input argument."
   "float __logf ( float x )

   Calculate the fast approximate base e logarithm of the input argument."
   "float __powf ( float x float y )

   Calculate the fast approximate of xy."
   "float __sinf ( float x )

   Calculate the fast approximate sine of the input argument."
   "float __tanf ( float x )

   Calculate the fast approximate tangent of the input argument."
   "double __dsqrt_rd ( double x )

   Compute √x in round-down mode."
   "double __dsqrt_rn ( double x )

   Compute √x in round-to-nearest-even mode."
   "double __dsqrt_ru ( double x )

   Compute √x in round-up mode."
   "double __dsqrt_rz ( double x )

   Compute √x in round-towards-zero mode."