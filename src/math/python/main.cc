/**
 * @file math/python/main.cc
 * @date Sat Mar 19 19:49:51 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Combines all modules to make up the complete bindings
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

#include "bob/core/python/ndarray.h"

void bind_math_lp_interiorpoint();
void bind_math_linsolve();
void bind_math_norminv();
void bind_math_stats();
void bind_math_histogram();
void bind_math_pavx();

BOOST_PYTHON_MODULE(_math) {

  bob::python::setup_python("bob mathematical classes and sub-classes");

  bind_math_lp_interiorpoint();
  bind_math_linsolve();
  bind_math_norminv();
  bind_math_stats();
  bind_math_histogram();
  bind_math_pavx();
}
