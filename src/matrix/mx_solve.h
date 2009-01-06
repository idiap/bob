#ifndef MX_SOLVE_INC
#define MX_SOLVE_INC

#include "Mat.h"
#include "Vec.h"

namespace Torch {

/*  Collection of factorisations functions.
    Based on the "Meschach Library", available at the
    anonymous ftp site thrain.anu.edu.au in the directory
    pub/meschach.

    @author David E. Stewart (david.stewart@anu.edu.au)
    @author Zbigniew Leyk (zbigniew.leyk@anu.edu.au)
    @author Ronan Collobert (collober@idiap.ch)
*/

/* Back substitution with optional over-riding diagonal.
    Can be in-situ but doesn't need to be. */
void mxUSolve(Mat * matrix, Vec * b, Vec * out, double diag);

// Forward elimination with (optional) default diagonal value.
void mxLSolve(Mat * matrix, Vec * b, Vec * out, double diag);

/* Forward elimination with (optional) default diagonal value.
    Use UPPER triangular part of matrix. */
void mxUTSolve(Mat * mat, Vec * b, Vec * out, double diag);

/* Solves Dx=b where D is the diagonal of A.
    May be in-situ. */
void mxDSolve(Mat * mat, Vec * b, Vec * x);

/* Back substitution with optional over-riding diagonal.
   Use the LOWER triangular part of matrix.
   Can be in-situ but doesn't need to be. */
void mxLTSolve(Mat * mat, Vec * b, Vec * out, double diag);


}

#endif
