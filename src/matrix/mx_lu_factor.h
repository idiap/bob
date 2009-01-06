#ifndef MX_LU_FACTOR_INC
#define MX_LU_FACTOR_INC

#include "Mat.h"
#include "Vec.h"
#include "Perm.h"

namespace Torch {

/** Collection of matrix factorisation operation functions.
    Based on the "Meschach Library", available at the
    anonymous ftp site thrain.anu.edu.au in the directory
    pub/meschach.

    Most matrix factorisation routines are in-situ
    unless otherwise specified.

    @author David E. Stewart (david.stewart@anu.edu.au)
    @author Zbigniew Leyk (zbigniew.leyk@anu.edu.au)
    @author Ronan Collobert (collober@idiap.ch)
*/
//@{

/** Gaussian elimination with scaled partial pivoting.
    -- Note: returns LU matrix which is #A#. */
void mxLUFactor(Mat * mat, Perm * pivot);

/**
   Given an LU factorisation in #A#, solve #Ax=b# */
void mxLUSolve(Mat * mat, Perm * pivot, Vec * b, Vec * x);

/**
   Given an LU factorisation in #A#, solve #A^T.x=b# */
void mxLUTSolve(Mat * mat, Perm * pivot, Vec * b, Vec * x);

/**
   Returns inverse of #A#, provided #A# is not too rank deficient.
   Uses LU factorisation. */
void mxInverse(Mat * mat, Mat * out);

/**
   Given #A# and #b#, solve #A.x=b# */
void mxSolve(Mat *mat, Vec *b, Vec *x);

//@}


}

#endif
