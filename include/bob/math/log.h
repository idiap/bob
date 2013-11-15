/**
 * @file bob/math/log.h
 * @date Fri Feb 10 20:02:07 2012 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_MATH_LOG_H
#define BOB_MATH_LOG_H

#include <cmath>
#include <limits>

namespace bob { namespace math {

/**
 * @ingroup MATH
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
