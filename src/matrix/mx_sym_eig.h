#ifndef MX_SYM_EIG_INC
#define MX_SYM_EIG_INC

#include "Vec.h"
#include "Mat.h"
#include "mx_givens.h"
#include "mx_hessenberg.h"

namespace Torch {

/**
	Routines for symmetric eigenvalue problems.

  Based on the "Meschach Library", available at the
  anonymous ftp site thrain.anu.edu.au in the directory
  pub/meschach.
  
  @author David E. Stewart (david.stewart@anu.edu.au)
  @author Zbigniew Leyk (zbigniew.leyk@anu.edu.au)
  @author Ronan Collobert (collober@idiap.ch)
*/
//@{

/** Finds eigenvalues of symmetric tridiagonal matrices.
    The matrix is represented by a pair of vectors #a# (diag entries)
		and #b# (sub-diag and super-diag entries).
    Eigenvalues in #a# on return, and eigenvectors in #mat_q#, if this one
    is not #NULL#.
*/
void mxTriEig(Vec * a, Vec * b, Mat * mat_q);

/** Computes eigenvalues of a dense symmetric matrix.
	  #mat_a# \emph{must} be symmetric on entry.
    Eigenvalues stored in #out#.
    #mat_q# contains orthogonal matrix of eigenvectors if not #NULL#.
*/
void mxSymEig(Mat * mat_a, Mat * mat_q, Vec * out);

//@}


}

#endif
