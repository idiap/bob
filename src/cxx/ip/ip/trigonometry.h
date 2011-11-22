/**
 * @file cxx/ip/ip/trigonometry.h
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef TRIGONOMETRY_INC
#define TRIGONOMETRY_INC

#include "core/general.h"

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
    \date
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
