/**
 * @file machine/python/jfa.cc
 * @date Sat Jul 23 21:41:15 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Python bindings for the JFA{Base,}Machine
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
#include <boost/shared_ptr.hpp>
#include <bob/core/python/ndarray.h>
#include <bob/machine/JFAMachine.h>
#include <bob/machine/GMMMachine.h>

using namespace boost::python;

static object py_getU(const bob::machine::JFABaseMachine& machine) {
  size_t n_CD = machine.getDimCD();
  size_t n_Ru = machine.getDimRu();
  bob::python::ndarray U(bob::core::array::t_float64, n_CD, n_Ru);
  blitz::Array<double,2> U_ = U.bz<double,2>();
  U_ = machine.getU();
  return U.self();
}

static void py_setU(bob::machine::JFABaseMachine& machine, bob::python::const_ndarray U) {
  const bob::core::array::typeinfo& info = U.type();
  if(info.dtype != bob::core::array::t_float64 || info.nd != 2)
    PYTHON_ERROR(TypeError, "cannot set array of type '%s'", info.str().c_str());
  const blitz::Array<double,2> U_ = U.bz<double,2>();
  machine.setU(U_);
}

static object py_getV(const bob::machine::JFABaseMachine& machine) {
  size_t n_CD = machine.getDimCD();
  size_t n_Rv = machine.getDimRv();
  bob::python::ndarray V(bob::core::array::t_float64, n_CD, n_Rv);
  blitz::Array<double,2> V_ = V.bz<double,2>();
  V_ = machine.getV();
  return V.self();
}

static void py_setV(bob::machine::JFABaseMachine& machine, bob::python::const_ndarray V) {
  const bob::core::array::typeinfo& info = V.type();
  if(info.dtype != bob::core::array::t_float64 || info.nd != 2)
    PYTHON_ERROR(TypeError, "cannot set array of type '%s'", info.str().c_str());
  const blitz::Array<double,2> V_ = V.bz<double,2>();
  machine.setV(V_);
}

static object py_getD(const bob::machine::JFABaseMachine& machine) {
  size_t n_CD = machine.getDimCD();
  bob::python::ndarray D(bob::core::array::t_float64, n_CD);
  blitz::Array<double,1> D_ = D.bz<double,1>();
  D_ = machine.getD();
  return D.self();
}

static void py_setD(bob::machine::JFABaseMachine& machine, bob::python::const_ndarray D) {
  const bob::core::array::typeinfo& info = D.type();
  if(info.dtype != bob::core::array::t_float64 || info.nd != 1)
    PYTHON_ERROR(TypeError, "cannot set array of type '%s'", info.str().c_str());
  const blitz::Array<double,1> D_ = D.bz<double,1>();
  machine.setD(D_);
}


static object py_getX(const bob::machine::JFAMachine& machine) {
  size_t n_Ru = machine.getDimRu();
  bob::python::ndarray X(bob::core::array::t_float64, n_Ru);
  blitz::Array<double,1> X_ = X.bz<double,1>();
  X_ = machine.getX();
  return X.self();
}

static object py_getY(const bob::machine::JFAMachine& machine) {
  size_t n_Rv = machine.getDimRv();
  bob::python::ndarray Y(bob::core::array::t_float64, n_Rv);
  blitz::Array<double,1> Y_ = Y.bz<double,1>();
  Y_ = machine.getY();
  return Y.self();
}

static void py_setY(bob::machine::JFAMachine& machine, bob::python::const_ndarray Y) {
  const bob::core::array::typeinfo& info = Y.type();
  if(info.dtype != bob::core::array::t_float64 || info.nd != 1)
    PYTHON_ERROR(TypeError, "cannot set array of type '%s'", info.str().c_str());
  const blitz::Array<double,1> Y_ = Y.bz<double,1>();
  machine.setY(Y_);
}

static object py_getZ(const bob::machine::JFAMachine& machine) {
  size_t n_CD = machine.getDimCD();
  bob::python::ndarray Z(bob::core::array::t_float64, n_CD);
  blitz::Array<double,1> Z_ = Z.bz<double,1>();
  Z_ = machine.getZ();
  return Z.self();
}

static void py_setZ(bob::machine::JFAMachine& machine, bob::python::const_ndarray Z) {
  const bob::core::array::typeinfo& info = Z.type();
  if(info.dtype != bob::core::array::t_float64 || info.nd != 1)
    PYTHON_ERROR(TypeError, "cannot set array of type '%s'", info.str().c_str());
  const blitz::Array<double,1> Z_ = Z.bz<double,1>();
  machine.setZ(Z_);
}

static void jfa_forward_list(bob::machine::JFAMachine& m, list stats, bob::python::ndarray score)
{
  // Extracts the vector of GMMStats from the python list
  int n_samples = len(stats);
  std::vector<boost::shared_ptr<const bob::machine::GMMStats> > gmm_stats;
  for(int s=0; s<n_samples; ++s)
  {
    boost::shared_ptr<bob::machine::GMMStats> gs = extract<boost::shared_ptr<bob::machine::GMMStats> >(stats[s]);
    gmm_stats.push_back(gs);
  }

  // Calls the forward function
  blitz::Array<double,1> score_ = score.bz<double,1>();
  m.forward(gmm_stats, score_);
}

static double jfa_forward_sample(bob::machine::JFAMachine& m, 
    const boost::shared_ptr<bob::machine::GMMStats> stats) {
  double score;
  // Calls the forward function
  m.forward(stats, score);
  return score;
}

 
static void jfa_estimateUx(bob::machine::JFAMachine& m, object stats, bob::python::ndarray Ux)
{
  // Extracts the vector of GMMStats from the python list
  boost::shared_ptr<bob::machine::GMMStats> gs = extract<boost::shared_ptr<bob::machine::GMMStats> >(stats);

  // Calls the forward function
  blitz::Array<double,1> Ux_ = Ux.bz<double,1>();
  m.estimateUx(gs, Ux_);
}

static double jfa_forward_sample_Ux(bob::machine::JFAMachine& m, object stats, bob::python::const_ndarray Ux)
{
  // Extracts the vector of GMMStats from the python list
  boost::shared_ptr<bob::machine::GMMStats> gs = extract<boost::shared_ptr<bob::machine::GMMStats> >(stats);

  // Calls the forward function
  double score;
  blitz::Array<double,1> Ux_ = Ux.bz<double,1>();
  m.forward(gs, Ux_, score);
  return score;
}



void bind_machine_jfa() 
{
  class_<bob::machine::JFABaseMachine, boost::shared_ptr<bob::machine::JFABaseMachine> >("JFABaseMachine", "A JFABaseMachine", init<boost::shared_ptr<bob::machine::GMMMachine>, optional<const size_t, const size_t> >((arg("ubm"), arg("ru")=1, arg("rv")=1), "Builds a new JFABaseMachine. A JFABaseMachine can be seen as a container for U, V and D when performing Joint Factor Analysis (JFA)."))
    .def(init<bob::io::HDF5File&>((arg("config")), "Constructs a new JFABaseMachine from a configuration file."))
    .def(init<const bob::machine::JFABaseMachine&>((arg("machine")), "Copy constructs a JFABaseMachine"))
    .def(self == self)
    .def("is_similar_to", &bob::machine::JFABaseMachine::is_similar_to, (arg("self"), arg("other"), arg("epsilon") = 1e-8), "Compares this JFABaseMachine with the 'other' one to be approximately the same; each parameter might differ maximal with the given epsilon.")
    .def("load", &bob::machine::JFABaseMachine::load, (arg("self"), arg("config")), "Loads the configuration parameters from a configuration file.")
    .def("save", &bob::machine::JFABaseMachine::save, (arg("self"), arg("config")), "Saves the configuration parameters to a configuration file.")
    .def("resize", &bob::machine::JFABaseMachine::resize, "Reset the dimensionality of the subspaces U and V.")
    .add_property("ubm", &bob::machine::JFABaseMachine::getUbm, &bob::machine::JFABaseMachine::setUbm)
    .add_property("u", &py_getU, &py_setU)
    .add_property("v", &py_getV, &py_setV)
    .add_property("d", &py_getD, &py_setD)
    .add_property("dim_c", &bob::machine::JFABaseMachine::getDimC)
    .add_property("dim_d", &bob::machine::JFABaseMachine::getDimD)
    .add_property("dim_cd", &bob::machine::JFABaseMachine::getDimCD)
    .add_property("dim_ru", &bob::machine::JFABaseMachine::getDimRu)
    .add_property("dim_rv", &bob::machine::JFABaseMachine::getDimRv)
  ;

  class_<bob::machine::JFAMachine, boost::shared_ptr<bob::machine::JFAMachine> >("JFAMachine", "A JFAMachine", init<const boost::shared_ptr<bob::machine::JFABaseMachine> >((arg("jfa_base")), "Builds a new JFAMachine. An attached JFABaseMachine should be provided for Joint Factor Analysis. The JFAMachine carries information about the speaker factors y and z, whereas a JFABaseMachine carries information about the matrices U, V and D."))
    .def(init<bob::io::HDF5File&>((arg("config")), "Constructs a new JFAMachine from a configuration file."))
    .def(init<const bob::machine::JFAMachine&>((arg("machine")), "Copy constructs a JFAMachine"))
    .def(self == self)
    .def("is_similar_to", &bob::machine::JFAMachine::is_similar_to, (arg("self"), arg("other"), arg("epsilon") = 1e-8), "Compares this JFAMachine with the 'other' one to be approximately the same; each parameter might differ maximal with the given epsilon.")
    .def("load", &bob::machine::JFAMachine::load, (arg("self"), arg("config")), "Loads the configuration parameters from a configuration file.")
    .def("save", &bob::machine::JFAMachine::save, (arg("self"), arg("config")), "Saves the configuration parameters to a configuration file.")
    .def("__call__", &jfa_forward_sample, (arg("self"), arg("gmm_stats")), "Processes GMM statistics and returns a score.")
    .def("forward", &jfa_forward_sample, (arg("self"), arg("gmm_stats")), "Processes GMM statistics and returns a score.")
    .def("estimate_ux", &jfa_estimateUx, (arg("self"), arg("gmm_stats")), "Processes GMM statistics to estimate Ux.")
    .def("__call__", &jfa_forward_list, (arg("self"), arg("gmm_stats"), arg("scores")), "Processes a list of GMM statistics and updates a score list.")
    .def("forward", &jfa_forward_list, (arg("self"), arg("gmm_stats"), arg("scores")), "Processes a list of GMM statistics and updates a score list.")
    .def("forward_ux", &jfa_forward_sample_Ux, (arg("self"), arg("gmm_stats"), arg("Ux")), "Processes GMM statistics and Ux to return a score.")
    .add_property("jfa_base", &bob::machine::JFAMachine::getJFABase, &bob::machine::JFAMachine::setJFABase)
    .add_property("x", &py_getX)
    .add_property("y", &py_getY, &py_setY)
    .add_property("z", &py_getZ, &py_setZ)
    .add_property("dim_c", &bob::machine::JFAMachine::getDimC)
    .add_property("dim_d", &bob::machine::JFAMachine::getDimD)
    .add_property("dim_cd", &bob::machine::JFAMachine::getDimCD)
    .add_property("dim_ru", &bob::machine::JFAMachine::getDimRu)
    .add_property("dim_rv", &bob::machine::JFAMachine::getDimRv)
  ;

}
