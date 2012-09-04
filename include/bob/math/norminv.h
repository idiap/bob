/**
 * @file cxx/math/math/norminv.h
 * @date Tue Apr 12 21:33:32 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file implements the inverse normal cumulative distribution function,
 * as described by Peter Acklam:
 *   http://home.online.no/~pjacklam/notes/invnorm/
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

#ifndef BOB_MATH_NORMINV_H
#define BOB_MATH_NORMINV_H

namespace bob {
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

#endif /* BOB_MATH_NORMINV_H */
