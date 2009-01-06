#ifndef PERM_OPERATIONS_INC
#define PERM_OPERATIONS_INC

#include "Perm.h"
#include "Mat.h"
#include "Vec.h"

namespace Torch {

/** Collection of permutations operation functions.
    Based on the "Meschach Library", available at the
    anonymous ftp site thrain.anu.edu.au in the directory
    pub/meschach.

    @author David E. Stewart (david.stewart@anu.edu.au)
    @author Zbigniew Leyk (zbigniew.leyk@anu.edu.au)
    @author Ronan Collobert (collober@idiap.ch)
*/
//@{
/** Invert permutation -- in situ.
    Taken from ACM Collected Algorithms 250. */
void mxPermInv(Perm * px, Perm * out);

/** Permutation multiplication (composition) -- not in-situ */
void mxPermMulPerm(Perm * px1, Perm * px2, Perm * out);

/** Permute vector -- can be in-situ */
void mxPermVec(Perm * px, Vec * vec, Vec * out);

/** Apply the inverse of px to x, returning the result in out.
    Can be in-situ, but "oh booooy!"... */
void mxPermInvVec(Perm * px, Vec * x, Vec * out);

/** Transpose elements of permutation.
    Really multiplying a permutation by a transposition.
    i1 and i2 are the elements to transpose. */
void mxTrPerm(Perm * px, int i1, int i2);

/** Permute columns of matrix A; out = A.px'. -- May NOT be in situ */
void mxPermColsMat(Perm * px, Mat * mat, Mat * out);

/** Permute rows of matrix A; out = px.A. -- May NOT be in situ */
void mxPermRowsMat(Perm * px, Mat * mat, Mat * out);

//@}


}

#endif
