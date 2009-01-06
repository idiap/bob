#ifndef MX_GIVENS_INC
#define MX_GIVENS_INC

#include "Vec.h"
#include "Mat.h"

namespace Torch {

/**
   Givens matrix operations routines.
   Routines for calculating and applying Givens rotations for/to
   vectors and also to matrices by row and by column.

   Based on the "Meschach Library", available at the
   anonymous ftp site thrain.anu.edu.au in the directory
   pub/meschach.

   @author David E. Stewart (david.stewart@anu.edu.au)
   @author Zbigniew Leyk (zbigniew.leyk@anu.edu.au)
   @author Ronan Collobert (collober@idiap.ch)
*/
//@{

/** Returns #c#, #s# parameters for Givens rotation
    to eliminate #y# in the vector #[ x y ]'#.
*/
void mx_givens(double x, double y, double * c, double * s);

/** Apply Givens rotation to #x#'s #i# and #k# components.
 */
void mx_rot_vec(Vec * x, int i, int k, double c, double s, Vec * out);

/** Premultiply #mat# by givens rotation described by #c#,#s#.
 */
void mx_rot_rows(Mat * mat, int i, int k, double c, double s, Mat * out);

/** Postmultiply #mat# by givens rotation described by #c#,#s#.
 */
void mx_rot_cols(Mat * mat, int i, int k, double c, double s, Mat * out);

//@}


}

#endif
