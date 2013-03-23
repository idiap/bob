/**
 * @file machine/python/machine.cc
 * @date Tue Jul 26 15:11:33 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
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
#include <boost/python.hpp>
#include <boost/concept_check.hpp>
#include <blitz/array.h>
#include <bob/machine/Machine.h>

#include <bob/core/python/ndarray.h>

using namespace boost::python;

static double forward(const bob::machine::Machine<blitz::Array<double,1>, double>& m,
    bob::python::const_ndarray input) {
  double output;
  m.forward(input.bz<double,1>(), output);
  return output;
}

static double forward_(const bob::machine::Machine<blitz::Array<double,1>, double>& m,
    bob::python::const_ndarray input) {
  double output;
  m.forward_(input.bz<double,1>(), output);
  return output;
}

void bind_machine_base() 
{
  class_<bob::machine::Machine<blitz::Array<double,1>, double>, boost::noncopyable>("MachineDoubleBase", 
      "Root class for all Machine<blitz::Array<double,1>, double>", no_init)
    .def("__call__", &forward_, (arg("self"), arg("input")), "Executes the machine on the given 1D numpy array of float64, and returns the output. NO CHECK is performed.")
    .def("forward", &forward, (arg("self"), arg("input")), "Executes the machine on the given 1D numpy array of float64, and returns the output.")
    .def("forward_", &forward_, (arg("self"), arg("input")), "Executes the machine on the given 1D numpy array of float64, and returns the output. NO CHECK is performed.")
  ;
}
