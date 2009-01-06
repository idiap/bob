#ifndef TRIGONOMETRY_INC
#define TRIGONOMETRY_INC

#include "general.h"

namespace Torch
{

#ifndef M_PI
#define M_PI           3.14159265358979323846  /* pi */
#endif
#ifndef M_PI_2
#define M_PI_2         1.57079632679489661923  /* pi/2 */
#endif
#ifndef M_PI_4
#define M_PI_4         0.78539816339744830962  /* pi/4 */
#endif

//-----------------------------------

/** @name radians to degrees convertion functions

    @author Sebastien Marcel (marcel@idiap.ch)
    @version 2.0
    \Date
    @since 1.0
*/
//@{
/// convert degrees to radian
double degree2radian(double d_);

/// convert radian2degrees
double radian2degree(double r_);
//@}

//-----------------------------------

}

#endif
