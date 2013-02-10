/**
 * @file bob/math/pavx.h
 * @date Sat Dec 8 19:35:25 2012 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Pool Adjacent Violators algorithm
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

#ifndef BOB_MATH_PAVX_H
#define BOB_MATH_PAVX_H

#include "bob/core/assert.h"
#include <utility>

namespace bob {
/**
 * \ingroup libmath_api
 * @{
 */
  namespace math {
   
    /**
      * @brief Pool Adjacent Violators algorithm. 
      *        Non-paramtetric optimization subject to monotonicity. 
      * This implementation is a C++ port of the Bosaris toolkit 
      * (utility_funcs/det/pavx.m) available at
      * https://sites.google.com/site/bosaristoolkit/
      * 
      * The code from bosaris is a simplified and/or adapted from the 
      * 'IsoMeans.m' code made available by Lutz Duembgen at:
      * http://www.imsv.unibe.ch/content/staff/personalhomepages/duembgen/software/isotonicregression/index_eng.html
      *
      * More formally, this code computes isotonic fits to a data vector y 
      * using a variant of the pool-adjacent-violators algorithm. In
      * particular, it minimizes the sum of (non-weighted) squared residuals 
      * using local means. The only difference/simplification compared to the
      * original code from Lutz Duembgen is that the data points of the 
      * vector y can not be weighted.
      */
    void pavx(const blitz::Array<double,1>& y, blitz::Array<double,1>& ghat);
    void pavx_(const blitz::Array<double,1>& y, blitz::Array<double,1>& ghat);
    /**
      * This variant additionally returns the width vector of the pav bins, 
      * from left to right (the number of bins is data dependent)
      */
    blitz::Array<size_t,1> pavxWidth(const blitz::Array<double,1>& y, blitz::Array<double,1>& ghat);
    /**
      * This variant additionally returns the width vector of the pav bins, 
      * from left to right (the number of bins is data dependent) as well as
      * the corresponding heights of bins (in increasing order).
      */
    std::pair<blitz::Array<size_t,1>, blitz::Array<double,1> > pavxWidthHeight(const blitz::Array<double,1>& y, blitz::Array<double,1>& ghat);
  }

/**
 * @}
 */
}

#endif /* BOB_MATH_PAVX_H */
