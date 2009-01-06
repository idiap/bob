#ifndef MX_HOUSEHOLDER_INC
#define MX_HOUSEHOLDER_INC

#include "Mat.h"
#include "Vec.h"

namespace Torch {

/**
   Householder transformation routines.
   Contains routines for calculating householder transformations,
   applying them to vectors and matrices by both row and column.

   Based on the "Meschach Library", available at the
   anonymous ftp site thrain.anu.edu.au in the directory
   pub/meschach.

   @author David E. Stewart (david.stewart@anu.edu.au)
   @author Zbigniew Leyk (zbigniew.leyk@anu.edu.au)
   @author Ronan Collobert (collober@idiap.ch)
*/
//@{

/** Calulates Householder vector.
    To eliminate all entries after the i0 entry of the vector vec.
    It is returned as out. May be in-situ.
*/
void mxHhVec(Vec * vec, int i0, double * beta, Vec * out, double * newval);


/** Apply Householder transformation to vector.
    May be in-situ. (#hh# is the Householder vector).
*/
void mxHhTrVec(Vec * hh, double beta, int i0, Vec * in, Vec * out);

/** Transform a matrix by a Householder vector by rows.
    Start at row i0 from column j0. In-situ.
*/
void mxHhTrRows(Mat * mat, int i0, int j0, Vec * hh, double beta);

/* Transform a matrix by a Householder vector by columns.
   Start at row i0 from column j0. In-situ.
*/
void mxHhTrCols(Mat * mat, int i0, int j0, Vec * hh, double beta);

//@}


}

#endif
