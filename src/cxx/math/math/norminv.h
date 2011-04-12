/**
 * @file src/cxx/math/math/norminv.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file implements the inverse normal cumulative distribution function,
 * as described by Peter Acklam:
 *   http://home.online.no/~pjacklam/notes/invnorm/
 * 
 */

#ifndef TORCH5SPRO_MATH_NORMINV_H
#define TORCH5SPRO_MATH_NORMINV_H

#include "core/array_assert.h"

namespace tca = Torch::core::array;

namespace Torch {
/**
 * \ingroup libmath_api
 * @{
 *
 */
  namespace math {

    /**
      * @brief Function which implements the inverse normal cumulative 
      *   distribution function, as described by Peter Acklam:
      *     http://home.online.no/~pjacklam/notes/invnorm/
      *   Please note that the boost library only provide such a feature
      *   since version 1.46.
      * @param p The argument (probability) of the inverse function. p should
      *   be in the range ]0.,1.[ strictly. The implementation even enforce a
      *   slightly narrower range.
      * @param mu The mean of the normal distribution
      * @param sigma The standard deviation of the normal distribution
      */
    double norminv(const double p, const double mu, const double sigma);

    /**
      * @brief Function which implements the inverse normal cumulative 
      *   distribution function (whose mean is 0 and standard deviation is 1), 
      *   as described by Peter Acklam:
      *     http://home.online.no/~pjacklam/notes/invnorm/
      *   Please note that the boost library only provide such a feature
      *   since version 1.46.
      * @param p The argument (probability) of the inverse function. p should
      *   be in the range ]0.,1.[ strictly. The implementation even enforce a
      *   slightly narrower range.
      * @warning The normal distribution is assumed to be zero mean and unit 
      *   variance.
      */
    double normsinv(const double p);
  }
/**
 * @}
 */
}

#endif /* TORCH5SPRO_MATH_NORMINV_H */
