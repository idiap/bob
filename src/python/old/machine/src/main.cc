/**
 * @file python/old/machine/src/main.cc
 * @date Wed Apr 6 14:49:40 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Combines all modules to make up the complete bindings
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

#include <boost/python.hpp>

using namespace boost::python;

void bind_machine_machine();
void bind_machine_ProbabilityDistribution();
void bind_machine_MultiVariateNormalDistribution();
void bind_machine_MultiVariateDiagonalGaussianDistribution();

BOOST_PYTHON_MODULE(libpytorch_machine) {
  scope().attr("__doc__") = "not available, nik lazy 44543543643";
  bind_machine_machine();
  bind_machine_ProbabilityDistribution();
  bind_machine_MultiVariateNormalDistribution();
  bind_machine_MultiVariateDiagonalGaussianDistribution();
}
