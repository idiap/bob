#ifndef TH_TENSOR_MATH_INC
#define TH_TENSOR_MATH_INC

#include "THTensor.h"

TH_API void THTensor_log(THTensor *tensor);
TH_API void THTensor_log1p(THTensor *tensor);
TH_API void THTensor_exp(THTensor *tensor);
TH_API void THTensor_cos(THTensor *tensor);
TH_API void THTensor_acos(THTensor *tensor);
TH_API void THTensor_cosh(THTensor *tensor);
TH_API void THTensor_sin(THTensor *tensor);
TH_API void THTensor_asin(THTensor *tensor);
TH_API void THTensor_sinh(THTensor *tensor);
TH_API void THTensor_tan(THTensor *tensor);
TH_API void THTensor_atan(THTensor *tensor);
TH_API void THTensor_tanh(THTensor *tensor);
TH_API void THTensor_pow(THTensor *tensor, double value);
TH_API void THTensor_sqrt(THTensor *tensor);
TH_API void THTensor_ceil(THTensor *tensor);
TH_API void THTensor_floor(THTensor *tensor);
TH_API void THTensor_abs(THTensor *tensor);

TH_API void THTensor_zero(THTensor *tensor);
TH_API void THTensor_add(THTensor *tensor, double value);
TH_API void THTensor_addTensor(THTensor *tensor, double value, THTensor *src);

TH_API void THTensor_mul(THTensor *tensor, double value);
TH_API void THTensor_cmul(THTensor *tensor, THTensor *src);
TH_API void THTensor_addcmul(THTensor *tensor, double value, THTensor *src1, THTensor *src2);
TH_API void THTensor_div(THTensor *tensor, double value);
TH_API void THTensor_cdiv(THTensor *tensor, THTensor *src);
TH_API void THTensor_addcdiv(THTensor *tensor, double value, THTensor *src1, THTensor *src2);
TH_API double THTensor_dot(THTensor *tensor, THTensor *src);

TH_API double THTensor_min(THTensor *tensor);
TH_API double THTensor_max(THTensor *tensor);
TH_API double THTensor_sum(THTensor *tensor);
TH_API double THTensor_mean(THTensor *tensor);
TH_API double THTensor_var(THTensor *tensor);
TH_API double THTensor_std(THTensor *tensor);
TH_API double THTensor_norm(THTensor *tensor, double value);
TH_API double THTensor_dist(THTensor *tensor, THTensor *src, double value);

TH_API void THTensor_addT2dotT1(THTensor *tensor, double alpha, THTensor *mat, THTensor *vec);
TH_API void THTensor_addT4dotT2(THTensor *tensor, double alpha, THTensor *t4, THTensor *t2);
TH_API void THTensor_addT1outT1(THTensor *tensor, double alpha, THTensor *vec1, THTensor *vec2);
TH_API void THTensor_addT2outT2(THTensor *tensor, double alpha, THTensor *m1, THTensor *m2);
TH_API void THTensor_addT2dotT2(THTensor *tensor, double alpha, THTensor *m1, THTensor *m2);

#endif
