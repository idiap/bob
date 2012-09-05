/**
 * @file python/machine/src/wiener.cc
 * @date Fri Sep 30 16:58:42 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Bindings for a WienerMachine
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
#include "machine/WienerMachine.h"

using namespace boost::python;
namespace mach = bob::machine;
namespace io = bob::io;

static blitz::Array<double,2> forward(const mach::WienerMachine& m,
    const blitz::Array<double,2>& input) {
  blitz::Array<double,2> output(input.shape());
  m.forward(input,output);
  return output;
}

static tuple get_shape(const mach::WienerMachine& m) {
  return make_tuple(m.getHeight(), m.getWidth());
}

static void set_shape(mach::WienerMachine& m, 
    const blitz::TinyVector<int,2>& s) {
  m.resize(s(0), s(1));
}

static void set_Ps(mach::WienerMachine& m, object o) {
  //try hard-core extraction - throws TypeError, if not possible
  blitz::Array<double,2> val = extract<blitz::Array<double,2> >(o);
  m.setPs(val);
}



void bind_machine_wiener() {
  class_<mach::WienerMachine, boost::shared_ptr<mach::WienerMachine> >("WienerMachine", "A Wiener filter.", init<size_t,size_t, const double, optional<const double> >((arg("height"), arg("width"), arg("pn"), arg("variance_threshold")=1e-8), "Constructs a new Wiener filter dedicated to images of the given dimensions. The filter is initialized with zero values."))
    .def(init<const blitz::Array<double,2>&, const double, optional<const double> >((arg("ps"), arg("pn"), arg("variance_threshold")=1e-8), "Constructs a new WienerMachine from a set of variance estimates ps, a noise level pn, and optionally a variance threshold level."))
    .def(init<io::HDF5File&>((arg("config")), "Constructs a new WienerMachine from a configuration file."))
    .def(init<>("Default constructor, builds a machine as with 'WienerMachine(0,0)'."))
    .def("load", &mach::WienerMachine::load, (arg("self"), arg("config")), "Loads the filter from a configuration file.")
    .def("save", &mach::WienerMachine::save, (arg("self"), arg("config")), "Saves the filter to a configuration file.")
    .add_property("pn", &mach::WienerMachine::getPn, &mach::WienerMachine::setPn)
    .add_property("variance_threshold", &mach::WienerMachine::getVarianceThreshold, &mach::WienerMachine::setVarianceThreshold)
    .add_property("ps", make_function(&mach::WienerMachine::getPs, return_internal_reference<>()), &set_Ps)
    .add_property("w", make_function(&mach::WienerMachine::getW, return_internal_reference<>()))
    .add_property("shape", &get_shape, &set_shape)
    .def("resize", &mach::WienerMachine::resize, (arg("self"), arg("height"), arg("width")), "Resizes the filter.")
    .def("__call__", &mach::WienerMachine::forward, (arg("self"), arg("input"), arg("output")), "Filters the input and saves results on the output.")
    .def("forward", &mach::WienerMachine::forward, (arg("self"), arg("input"), arg("output")), "Filters the input and saves results on the output.")
    .def("__call__", &forward, (arg("self"), arg("input")), "Filters the input and returns the output. This method implies in copying out the output data and is, therefore, less efficient as its counterpart that sets the output given as parameter. If you have to do a tight loop, consider using that variant instead of this one.")
    .def("forward", &forward, (arg("self"), arg("input")), "Filter the input and returns the output. This method implies in copying out the output data and is, therefore, less efficient as its counterpart that sets the output given as parameter. If you have to do a tight loop, consider using that variant instead of this one.")
    ;
}
