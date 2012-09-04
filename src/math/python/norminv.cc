/**
 * @file python/math/src/norminv.cc
 * @date Wed Apr 13 09:20:40 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds the inverse normal cumulative distribution into python
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

#include <boost/python.hpp>

#include "bob/math/norminv.h"

using namespace boost::python;

static const char* NORMSINV_DOC = "Compute the inverse normal cumulative distribution for a probability p, given a distribution with zero mean and and unit variance.";
static const char* NORMINV_DOC = "Compute the inverse normal cumulative distribution for a probability p, given a distribution with mean mu and standard deviation sigma.";

void bind_math_norminv()
{
  def("normsinv", (double (*)(const double))&bob::math::normsinv, (arg("p")), NORMSINV_DOC);
  def("norminv", (double (*)(const double, const double, const double))&bob::math::norminv, (arg("p"), arg("mu"), arg("sigma")), NORMINV_DOC);
}

