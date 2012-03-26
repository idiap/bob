/**
 * @file python/machine/src/jfa.cc
 * @date Sat Jul 23 21:41:15 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Python bindings for the JFA{Base,}Machine
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
#include <boost/shared_ptr.hpp>
#include "core/python/ndarray.h"
#include "machine/JFAMachine.h"
#include "machine/GMMMachine.h"

using namespace boost::python;
namespace mach = bob::machine;
namespace ca = bob::core::array;
namespace io = bob::io;
namespace tp = bob::python;

static object py_getU(const mach::JFABaseMachine& machine) {
  size_t n_CD = machine.getDimCD();
  size_t n_Ru = machine.getDimRu();
  tp::ndarray U(ca::t_float64, n_CD, n_Ru);
  blitz::Array<double,2> U_ = U.bz<double,2>();
  U_ = machine.getU();
  return U.self();
}

static void py_setU(mach::JFABaseMachine& machine, tp::const_ndarray U) {
  const ca::typeinfo& info = U.type();
  if(info.dtype != ca::t_float64 || info.nd != 2)
    PYTHON_ERROR(TypeError, "cannot set array of type '%s'", info.str().c_str());
  const blitz::Array<double,2> U_ = U.bz<double,2>();
  machine.setU(U_);
}

static object py_getV(const mach::JFABaseMachine& machine) {
  size_t n_CD = machine.getDimCD();
  size_t n_Rv = machine.getDimRv();
  tp::ndarray V(ca::t_float64, n_CD, n_Rv);
  blitz::Array<double,2> V_ = V.bz<double,2>();
  V_ = machine.getV();
  return V.self();
}

static void py_setV(mach::JFABaseMachine& machine, tp::const_ndarray V) {
  const ca::typeinfo& info = V.type();
  if(info.dtype != ca::t_float64 || info.nd != 2)
    PYTHON_ERROR(TypeError, "cannot set array of type '%s'", info.str().c_str());
  const blitz::Array<double,2> V_ = V.bz<double,2>();
  machine.setV(V_);
}

static object py_getD(const mach::JFABaseMachine& machine) {
  size_t n_CD = machine.getDimCD();
  tp::ndarray D(ca::t_float64, n_CD);
  blitz::Array<double,1> D_ = D.bz<double,1>();
  D_ = machine.getD();
  return D.self();
}

static void py_setD(mach::JFABaseMachine& machine, tp::const_ndarray D) {
  const ca::typeinfo& info = D.type();
  if(info.dtype != ca::t_float64 || info.nd != 1)
    PYTHON_ERROR(TypeError, "cannot set array of type '%s'", info.str().c_str());
  const blitz::Array<double,1> D_ = D.bz<double,1>();
  machine.setD(D_);
}


static object py_getX(const mach::JFAMachine& machine) {
  size_t n_Ru = machine.getDimRu();
  tp::ndarray X(ca::t_float64, n_Ru);
  blitz::Array<double,1> X_ = X.bz<double,1>();
  X_ = machine.getX();
  return X.self();
}

static object py_getY(const mach::JFAMachine& machine) {
  size_t n_Rv = machine.getDimRv();
  tp::ndarray Y(ca::t_float64, n_Rv);
  blitz::Array<double,1> Y_ = Y.bz<double,1>();
  Y_ = machine.getY();
  return Y.self();
}

static void py_setY(mach::JFAMachine& machine, tp::const_ndarray Y) {
  const ca::typeinfo& info = Y.type();
  if(info.dtype != ca::t_float64 || info.nd != 1)
    PYTHON_ERROR(TypeError, "cannot set array of type '%s'", info.str().c_str());
  const blitz::Array<double,1> Y_ = Y.bz<double,1>();
  machine.setY(Y_);
}

static object py_getZ(const mach::JFAMachine& machine) {
  size_t n_CD = machine.getDimCD();
  tp::ndarray Z(ca::t_float64, n_CD);
  blitz::Array<double,1> Z_ = Z.bz<double,1>();
  Z_ = machine.getZ();
  return Z.self();
}

static void py_setZ(mach::JFAMachine& machine, tp::const_ndarray Z) {
  const ca::typeinfo& info = Z.type();
  if(info.dtype != ca::t_float64 || info.nd != 1)
    PYTHON_ERROR(TypeError, "cannot set array of type '%s'", info.str().c_str());
  const blitz::Array<double,1> Z_ = Z.bz<double,1>();
  machine.setZ(Z_);
}

static void jfa_forward_list(mach::JFAMachine& m, list stats, tp::ndarray score)
{
  // Extracts the vector of GMMStats from the python list
  int n_samples = len(stats);
  std::vector<boost::shared_ptr<const bob::machine::GMMStats> > gmm_stats;
  for(int s=0; s<n_samples; ++s)
  {
    boost::shared_ptr<mach::GMMStats> gs = extract<boost::shared_ptr<mach::GMMStats> >(stats[s]);
    gmm_stats.push_back(gs);
  }

  // Calls the forward function
  blitz::Array<double,1> score_ = score.bz<double,1>();
  m.forward(gmm_stats, score_);
}

static double jfa_forward_sample(mach::JFAMachine& m, 
    const boost::shared_ptr<bob::machine::GMMStats> stats) {
  double score;
  // Calls the forward function
  m.forward(stats, score);
  return score;
}

void bind_machine_jfa() 
{
  class_<mach::JFABaseMachine, boost::shared_ptr<mach::JFABaseMachine> >("JFABaseMachine", "A JFABaseMachine", init<boost::shared_ptr<mach::GMMMachine>, optional<const size_t, const size_t> >((arg("ubm"), arg("ru")=1, arg("rv")=1), "Builds a new JFABaseMachine. A JFABaseMachine can be seen as a container for U, V and D when performing Joint Factor Analysis (JFA)."))
    .def(init<io::HDF5File&>((arg("config")), "Constructs a new JFABaseMachine from a configuration file."))
    .def(init<const mach::JFABaseMachine&>((arg("machine")), "Copy constructs a JFABaseMachine"))
    .def(self == self)
    .def("load", &mach::JFABaseMachine::load, (arg("self"), arg("config")), "Loads the configuration parameters from a configuration file.")
    .def("save", &mach::JFABaseMachine::save, (arg("self"), arg("config")), "Saves the configuration parameters to a configuration file.")
    .def("resize", &mach::JFABaseMachine::resize, "Reset the dimensionality of the subspaces U and V.")
    .add_property("ubm", &mach::JFABaseMachine::getUbm, &mach::JFABaseMachine::setUbm)
    .add_property("u", &py_getU, &py_setU)
    .add_property("v", &py_getV, &py_setV)
    .add_property("d", &py_getD, &py_setD)
    .add_property("dim_c", &mach::JFABaseMachine::getDimC)
    .add_property("dim_d", &mach::JFABaseMachine::getDimD)
    .add_property("dim_cd", &mach::JFABaseMachine::getDimCD)
    .add_property("dim_ru", &mach::JFABaseMachine::getDimRu)
    .add_property("dim_rv", &mach::JFABaseMachine::getDimRv)
  ;

  class_<mach::JFAMachine, boost::shared_ptr<mach::JFAMachine> >("JFAMachine", "A JFAMachine", init<const boost::shared_ptr<mach::JFABaseMachine> >((arg("jfa_base")), "Builds a new JFAMachine. An attached JFABaseMachine should be provided for Joint Factor Analysis. The JFAMachine carries information about the speaker factors y and z, whereas a JFABaseMachine carries information about the matrices U, V and D."))
    .def(init<io::HDF5File&>((arg("config")), "Constructs a new JFAMachine from a configuration file."))
    .def(init<const mach::JFAMachine&>((arg("machine")), "Copy constructs a JFAMachine"))
    .def(self == self)
    .def("load", &mach::JFAMachine::load, (arg("self"), arg("config")), "Loads the configuration parameters from a configuration file.")
    .def("save", &mach::JFAMachine::save, (arg("self"), arg("config")), "Saves the configuration parameters to a configuration file.")
    .def("__call__", &jfa_forward_sample, (arg("self"), arg("gmm_stats")), "Processes GMM statistics and returns a score.")
    .def("forward", &jfa_forward_sample, (arg("self"), arg("gmm_stats")), "Processes GMM statistics and returns a score.")
    .def("__call__", &jfa_forward_list, (arg("self"), arg("gmm_stats"), arg("scores")), "Processes a list of GMM statistics and updates a score list.")
    .def("forward", &jfa_forward_list, (arg("self"), arg("gmm_stats"), arg("scores")), "Processes a list of GMM statistics and updates a score list.")
    .add_property("jfa_base", &mach::JFAMachine::getJFABase, &mach::JFAMachine::setJFABase)
    .add_property("x", &py_getX)
    .add_property("y", &py_getY, &py_setY)
    .add_property("z", &py_getZ, &py_setZ)
    .add_property("dim_c", &mach::JFAMachine::getDimC)
    .add_property("dim_d", &mach::JFAMachine::getDimD)
    .add_property("dim_cd", &mach::JFAMachine::getDimCD)
    .add_property("dim_ru", &mach::JFAMachine::getDimRu)
    .add_property("dim_rv", &mach::JFAMachine::getDimRv)
  ;

}
