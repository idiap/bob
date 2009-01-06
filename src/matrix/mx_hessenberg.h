#ifndef MX_HESSENBERG_INC
#define MX_HESSENBERG_INC

#include "Mat.h"
#include "Vec.h"

namespace Torch {

/**
   Routines for determining Hessenberg factorisations.
   
   Based on the "Meschach Library", available at the
   anonymous ftp site thrain.anu.edu.au in the directory
   pub/meschach.
   
   @author David E. Stewart (david.stewart@anu.edu.au)
   @author Zbigniew Leyk (zbigniew.leyk@anu.edu.au)
   @author Ronan Collobert (collober@idiap.ch)
*/
//@{

/** Compute Hessenberg factorisation in compact form.
   Factorisation performed in situ.
*/
void mxHFactor(Mat * mat, Vec * diag, Vec * beta);


/** Construct the Hessenberg orthogonalising matrix Q.
    i.e. Hess M = Q.M.Q'.
*/
void mxMakeHQ(Mat * h_mat, Vec * diag, Vec * beta, Mat * q_out);

/** Construct actual Hessenberg matrix.
 */
void mxMakeH(Mat * h_mat, Mat * h_out);

//@}


}

#endif
