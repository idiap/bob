/**
 * @file bob/math/log.h
 * @date Fri Feb 10 20:02:07 2012 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
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

#ifndef BOB_MATH_LOG_H
#define BOB_MATH_LOG_H

#include <cmath>
#include <limits>

namespace bob { namespace math {

/**
 * @brief Some logarithm constants and functions inherited from Torch3. 
 *   It seems that the 'magic' threshold MINUS_LOG_THRESHOLD is also 
 *   defined in a similar way in the PLearn library 
 *   (http://plearn.berlios.de/). I have no clue about the history of the
 *   following.
 */
namespace Log
{
  #define MINUS_LOG_THRESHOLD -39.14
  const double LogZero = -std::numeric_limits<double>::max();
  const double LogOne = 0.;
  const double Log2Pi = log(2*M_PI);
  
  double logAdd(double log_a, double log_b);
  double logSub(double log_a, double log_b);
}

}}
#endif
