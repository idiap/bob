#ifndef MAT_OPERATIONS_INC
#define MAT_OPERATIONS_INC

#include "Mat.h"
#include "Vec.h"

namespace Torch {

/** Collection of matrix operation functions.
    Based on the "Meschach Library", available at the
    anonymous ftp site thrain.anu.edu.au in the directory
    pub/meschach.

    @author David E. Stewart (david.stewart@anu.edu.au)
    @author Zbigniew Leyk (zbigniew.leyk@anu.edu.au)
    @author Ronan Collobert (collober@idiap.ch)
*/
//@{
/// Matrix addition -- may be in-situ
void mxMatAddMat(Mat * mat1, Mat * mat2, Mat * out);
/// Matrix subtraction -- may be in-situ
void mxMatSubMat(Mat * mat1, Mat * mat2, Mat * out);
/// Matrix-matrix multiplication
void mxMatMulMat(Mat * mat1, Mat * mat2, Mat * out);
/** Matrix-matrix transposed multiplication.
    -- #A.B^T# is stored in out */
void mxMatMulTrMat(Mat * mat1, Mat * mat2, Mat * out);
/** Matrix transposed-matrix multiplication.
    -- #A^T.B# is stored in out */
void mxTrMatMulMat(Mat * mat1, Mat * mat2, Mat * out);
/** Matrix-vector multiplication.
    -- Note: #b# is treated as a column vector */
void mxMatMulVec(Mat * mat, Vec * b, Vec * out);
/// Scalar-matrix multiply -- may be in-situ
void mxDoubleMulMat(double scalar, Mat * matrix, Mat * out);
/** Vector-matrix multiplication.
    -- Note: #b# is treated as a row vector */
void mxVecMulMat(Vec * b, Mat * mat, Vec * out);
/// Transpose matrix
void mxTrMat(Mat * in, Mat * out);
/** Swaps rows i and j of matrix A upto column lim.
    #lo# and #hi# to -1 if you want to swap all */
void mxSwapRowsMat(Mat * mat, int i, int j, int lo, int hi);
/** Swap columns i and j of matrix A upto row lim.
    #lo# and #hi# to -1 if you want to swap all */
void mxSwapColsMat(Mat * mat, int i, int j, int lo, int hi);
/** Matrix-scalar multiply and add.
    -- may be in situ.
    -- returns out == A1 + s*A2 */
void mxMatAddDoubleMulMat(Mat * mat1, Mat * mat2, double s, Mat * out);
/** Matrix-vector multiply and add.
    -- may not be in situ
    -- returns out == v1 + alpha*A*v2 */
void mxVecAddDoubleMulMatMulVec(Vec * v1, double alpha, Mat * mat, Vec * v2,
			      Vec * out);
/** Vector-matrix multiply and add
    -- may not be in situ
    -- returns out' == v1' + alpha * v2'*A */
void mxVecAddDoubleMulVecMulMat(Vec * v1, double alpha, Vec * v2, Mat * mat,
			      Vec * out);
//@}


}

#endif
