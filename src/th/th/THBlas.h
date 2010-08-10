#ifndef TH_BLAS_INC
#define TH_BLAS_INC

#include "THGeneral.h"

/* Level 1 */

TH_API void THBlas_swap(long size, double *x, long xStride, double *y, long yStride);
TH_API void THBlas_scale(long size, double alpha, double *y, long yStride);
TH_API void THBlas_copy(long size, const double *x, long xStride, double *y, long yStride);
TH_API void THBlas_add(long size, double alpha, double *x, long xStride, double *y, long yStride);
TH_API double THBlas_dot(long size, double *x, long xStride, double *y, long yStride);

/* Level 2 */

TH_API void THBlas_matVec(int trans, long nRow, long nColumn, double alpha, 
                          double *m, long mStride, double *x, long xStride, 
                          double beta, double *y, long yStride);

TH_API void THBlas_outerProduct(long nRow, long nColumn, double alpha, 
                                double *x, long xStride, double *y, long yStride, 
                                double *m, long mStride);

/* Level 3 */

TH_API void THBlas_matMat(int transA, int transB, 
                          long nRow, long nColumn, long nRC, double alpha, 
                          double *a, long aStride, double *b, long bStride, 
                          double beta, double *c, long cStride);

#endif
