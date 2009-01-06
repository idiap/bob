#ifndef MX_LOW_LEVEL_INC
#define MX_LOW_LEVEL_INC

#include "general.h"

namespace Torch {
#define VUNROLL

/*  Collection of low level functions.
    Based on the "Meschach Library", available at the
    anonymous ftp site thrain.anu.edu.au in the directory
    pub/meschach.

    @author David E. Stewart (david.stewart@anu.edu.au)
    @author Zbigniew Leyk (zbigniew.leyk@anu.edu.au)
    @author Ronan Collobert (collober@idiap.ch)
*/

// Inner product
double mxIp__(double * dp1, double * dp2, int len);
// Scalar multiply and add c.f. v_mltadd()
void mxDoubleMulAdd__(double * dp1, double * dp2, double s, int len);
// Scalar multiply array c.f. sv_mlt()
void mxDoubleMul__(double * dp, double s, double * out, int len);
// Add arrays c.f. v_add()
void mxAdd__(double * dp1, double * dp2, double * out, int len);
// Subtract arrays c.f. v_sub()
void mxSub__(double * dp1, double * dp2, double * out, int len);
// Zeros an array of floating point numbers
void mxZero__(double * dp, int len);


}

#endif
