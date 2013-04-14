/**
 * @file machine/python/jfa.cc
 * @date Sat Jul 23 21:41:15 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Python bindings for the FA-related machines
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
#include <bob/core/python/ndarray.h>
#include <boost/shared_ptr.hpp>
#include <bob/machine/JFAMachine.h>
#include <bob/machine/GMMMachine.h>

using namespace boost::python;

static object py_jfa_getU(const bob::machine::JFABase& machine) 
{
  const blitz::Array<double,2>& Um = machine.getU();
  bob::python::ndarray U(bob::core::array::t_float64, Um.extent(0), Um.extent(1));
  blitz::Array<double,2> U_ = U.bz<double,2>();
  U_ = Um;
  return U.self();
}

static void py_jfa_setU(bob::machine::JFABase& machine, 
  bob::python::const_ndarray U) 
{
  const blitz::Array<double,2> U_ = U.bz<double,2>();
  machine.setU(U_);
}

static object py_jfa_getV(const bob::machine::JFABase& machine) 
{
  const blitz::Array<double,2>& Vm = machine.getV();
  bob::python::ndarray V(bob::core::array::t_float64, Vm.extent(0), Vm.extent(1));
  blitz::Array<double,2> V_ = V.bz<double,2>();
  V_ = Vm;
  return V.self();
}

static void py_jfa_setV(bob::machine::JFABase& machine,
  bob::python::const_ndarray V) 
{
  const blitz::Array<double,2> V_ = V.bz<double,2>();
  machine.setV(V_);
}

static object py_jfa_getD(const bob::machine::JFABase& machine) 
{
  const blitz::Array<double,1>& Dm = machine.getD();
  bob::python::ndarray D(bob::core::array::t_float64, Dm.extent(0));
  blitz::Array<double,1> D_ = D.bz<double,1>();
  D_ = Dm;
  return D.self();
}

static void py_jfa_setD(bob::machine::JFABase& machine,
  bob::python::const_ndarray D)
{
  const blitz::Array<double,1> D_ = D.bz<double,1>();
  machine.setD(D_);
}

static object py_jfa_getX(const bob::machine::JFAMachine& machine) {
  const blitz::Array<double,1>& Xm = machine.getX();
  bob::python::ndarray X(bob::core::array::t_float64, Xm.extent(0));
  blitz::Array<double,1> X_ = X.bz<double,1>();
  X_ = Xm;
  return X.self();
}

static object py_jfa_getY(const bob::machine::JFAMachine& machine) {
  const blitz::Array<double,1>& Ym = machine.getY();
  bob::python::ndarray Y(bob::core::array::t_float64, Ym.extent(0));
  blitz::Array<double,1> Y_ = Y.bz<double,1>();
  Y_ = Ym;
  return Y.self();
}

static void py_jfa_setY(bob::machine::JFAMachine& machine, bob::python::const_ndarray Y) {
  const blitz::Array<double,1>& Y_ = Y.bz<double,1>();
  machine.setY(Y_);
}

static object py_jfa_getZ(const bob::machine::JFAMachine& machine) {
  const blitz::Array<double,1>& Zm = machine.getZ();
  bob::python::ndarray Z(bob::core::array::t_float64, Zm.extent(0));
  blitz::Array<double,1> Z_ = Z.bz<double,1>();
  Z_ = Zm;
  return Z.self();
}

static void py_jfa_setZ(bob::machine::JFAMachine& machine, bob::python::const_ndarray Z) {
  const blitz::Array<double,1> Z_ = Z.bz<double,1>();
  machine.setZ(Z_);
}

static void py_jfa_estimateX(bob::machine::JFAMachine& machine, 
  const bob::machine::GMMStats& gmm_stats, bob::python::ndarray x)
{
  blitz::Array<double,1> x_ = x.bz<double,1>();
  machine.estimateX(gmm_stats, x_);
}

static void py_jfa_estimateUx(bob::machine::JFAMachine& machine, 
  const bob::machine::GMMStats& gmm_stats, bob::python::ndarray ux)
{
  blitz::Array<double,1> ux_ = ux.bz<double,1>();
  machine.estimateUx(gmm_stats, ux_);
}

static double py_jfa_forwardUx(bob::machine::JFAMachine& machine, 
  const bob::machine::GMMStats& gmm_stats, bob::python::const_ndarray ux)
{
  const blitz::Array<double,1> ux_ = ux.bz<double,1>();
  double score;
  machine.forward(gmm_stats, ux_, score);
  return score;
}



static object py_isv_getU(const bob::machine::ISVBase& machine) 
{
  const blitz::Array<double,2>& Um = machine.getU();
  bob::python::ndarray U(bob::core::array::t_float64, Um.extent(0), Um.extent(1));
  blitz::Array<double,2> U_ = U.bz<double,2>();
  U_ = Um;
  return U.self();
}

static void py_isv_setU(bob::machine::ISVBase& machine, 
  bob::python::const_ndarray U) 
{
  const blitz::Array<double,2> U_ = U.bz<double,2>();
  machine.setU(U_);
}

static object py_isv_getD(const bob::machine::ISVBase& machine) 
{
  const blitz::Array<double,1>& Dm = machine.getD();
  bob::python::ndarray D(bob::core::array::t_float64, Dm.extent(0));
  blitz::Array<double,1> D_ = D.bz<double,1>();
  D_ = Dm;
  return D.self();
}

static void py_isv_setD(bob::machine::ISVBase& machine,
  bob::python::const_ndarray D)
{
  const blitz::Array<double,1> D_ = D.bz<double,1>();
  machine.setD(D_);
}

static object py_isv_getX(const bob::machine::ISVMachine& machine) {
  const blitz::Array<double,1>& Xm = machine.getX();
  bob::python::ndarray X(bob::core::array::t_float64, Xm.extent(0));
  blitz::Array<double,1> X_ = X.bz<double,1>();
  X_ = Xm;
  return X.self();
}

static object py_isv_getZ(const bob::machine::ISVMachine& machine) {
  const blitz::Array<double,1>& Zm = machine.getZ();
  bob::python::ndarray Z(bob::core::array::t_float64, Zm.extent(0));
  blitz::Array<double,1> Z_ = Z.bz<double,1>();
  Z_ = Zm;
  return Z.self();
}

static void py_isv_setZ(bob::machine::ISVMachine& machine, bob::python::const_ndarray Z) {
  const blitz::Array<double,1> Z_ = Z.bz<double,1>();
  machine.setZ(Z_);
}

static void py_isv_estimateX(bob::machine::ISVMachine& machine, 
  const bob::machine::GMMStats& gmm_stats, bob::python::ndarray x)
{
  blitz::Array<double,1> x_ = x.bz<double,1>();
  machine.estimateX(gmm_stats, x_);
}

static void py_isv_estimateUx(bob::machine::ISVMachine& machine, 
  const bob::machine::GMMStats& gmm_stats, bob::python::ndarray ux)
{
  blitz::Array<double,1> ux_ = ux.bz<double,1>();
  machine.estimateUx(gmm_stats, ux_);
}

static double py_isv_forwardUx(bob::machine::ISVMachine& machine, 
  const bob::machine::GMMStats& gmm_stats, bob::python::const_ndarray ux)
{
  const blitz::Array<double,1> ux_ = ux.bz<double,1>();
  double score;
  machine.forward(gmm_stats, ux_, score);
  return score;
}


static double py_gen1_forward(const bob::machine::Machine<bob::machine::GMMStats, double>& m,
  const bob::machine::GMMStats& stats)
{
  double output;
  m.forward(stats, output);
  return output;
}

static double py_gen1_forward_(const bob::machine::Machine<bob::machine::GMMStats, double>& m, 
  const bob::machine::GMMStats& stats)
{
  double output;
  m.forward_(stats, output);
  return output;
}

static void py_gen2b_forward(const bob::machine::Machine<bob::machine::GMMStats, blitz::Array<double,1> >& m,
  const bob::machine::GMMStats& stats, bob::python::const_ndarray output)
{
  blitz::Array<double,1> output_ = output.bz<double,1>();
  m.forward(stats, output_);
}

static void py_gen2b_forward_(const bob::machine::Machine<bob::machine::GMMStats, blitz::Array<double,1> >& m,
  const bob::machine::GMMStats& stats, bob::python::const_ndarray output)
{
  blitz::Array<double,1> output_ = output.bz<double,1>();
  m.forward_(stats, output_);
}


void bind_machine_jfa() 
{
  class_<bob::machine::Machine<bob::machine::GMMStats, double>, boost::noncopyable>("MachineGMMStatsScalarBase", 
      "Root class for all Machine<bob::machine::GMMStats, double>", no_init)
    .def("__call__", &py_gen1_forward_, (arg("self"), arg("input")), "Executes the machine on the GMMStats, and returns the (scalar) output. NO CHECK is performed.")
    .def("forward", &py_gen1_forward, (arg("self"), arg("input")), "Executes the machine on the GMMStats, and returns the (scalar) output.")
    .def("forward_", &py_gen1_forward_, (arg("self"), arg("input")), "Executes the machine on the GMMStats, and returns the (scalar) output. NO CHECK is performed.")
  ;

  class_<bob::machine::Machine<bob::machine::GMMStats, blitz::Array<double,1> >, boost::noncopyable>("MachineGMMStatsA1DBase", 
      "Root class for all Machine<bob::machine::GMMStats, blitz::Array<double,1>", no_init)
    .def("__call__", &py_gen2b_forward_, (arg("self"), arg("input"), arg("output")), "Executes the machine on the GMMStats, and returns the (scalar) output. NO CHECK is performed.")
    .def("forward", &py_gen2b_forward, (arg("self"), arg("input"), arg("output")), "Executes the machine on the GMMStats, and returns the (scalar) output.")
    .def("forward_", &py_gen2b_forward_, (arg("self"), arg("input"), arg("output")), "Executes the machine on the GMMStats, and returns the (scalar) output. NO CHECK is performed.")
  ;


  class_<bob::machine::JFABase, boost::shared_ptr<bob::machine::JFABase>, bases<bob::machine::Machine<bob::machine::GMMStats, double> > >("JFABase", "A JFABase A JFABase instance can be seen as a container for U, V and D when performing Joint Factor Analysis (JFA). TODO: add references", init<const boost::shared_ptr<bob::machine::GMMMachine>, optional<const size_t, const size_t> >((arg("ubm"), arg("ru")=1, arg("rv")=1), "Builds a new JFABase."))
    .def(init<>("Constructs a 1x1 JFABase instance. You have to set a UBM GMM and resize the U, V and D subspaces afterwards."))
    .def(init<bob::io::HDF5File&>((arg("config")), "Constructs a new JFABaseMachine from a configuration file."))
    .def(init<const bob::machine::JFABase&>((arg("machine")), "Copy constructs a JFABase"))
    .def(self == self)
    .def(self != self)
    .def("is_similar_to", &bob::machine::JFABase::is_similar_to, (arg("self"), arg("other"), arg("r_epsilon")=1e-5, arg("a_epsilon")=1e-8), "Compares this JFABase with the 'other' one to be approximately the same.")
    .def("load", &bob::machine::JFABase::load, (arg("self"), arg("config")), "Loads the configuration parameters from a configuration file.")
    .def("save", &bob::machine::JFABase::save, (arg("self"), arg("config")), "Saves the configuration parameters to a configuration file.")
    .def("resize", &bob::machine::JFABase::resize, (arg("self"), arg("ru"), arg("rv")), "Reset the dimensionality of the subspaces U and V.")
    .add_property("ubm", &bob::machine::JFABase::getUbm, &bob::machine::JFABase::setUbm)
    .add_property("u", &py_jfa_getU, &py_jfa_setU)
    .add_property("v", &py_jfa_getV, &py_jfa_setV)
    .add_property("d", &py_jfa_getD, &py_jfa_setD)
    .add_property("dim_c", &bob::machine::JFABase::getDimC)
    .add_property("dim_d", &bob::machine::JFABase::getDimD)
    .add_property("dim_cd", &bob::machine::JFABase::getDimCD)
    .add_property("dim_ru", &bob::machine::JFABase::getDimRu)
    .add_property("dim_rv", &bob::machine::JFABase::getDimRv)
  ;

  class_<bob::machine::JFAMachine, boost::shared_ptr<bob::machine::JFAMachine>, bases<bob::machine::Machine<bob::machine::GMMStats, double> > >("JFAMachine", "A JFAMachine. An attached JFABase should be provided for Joint Factor Analysis. The JFAMachine carries information about the speaker factors y and z, whereas a JFABase carries information about the matrices U, V and D.", init<const boost::shared_ptr<bob::machine::JFABase> >((arg("jfa_base")), "Builds a new JFAMachine."))
    .def(init<>("Constructs a 1x1 JFAMachine instance. You have to set a JFABase afterwards."))
    .def(init<bob::io::HDF5File&>((arg("config")), "Constructs a new JFAMachine from a configuration file."))
    .def(init<const bob::machine::JFAMachine&>((arg("machine")), "Copy constructs a JFAMachine"))
    .def(self == self)
    .def(self != self)
    .def("is_similar_to", &bob::machine::JFAMachine::is_similar_to, (arg("self"), arg("other"), arg("r_epsilon")=1e-5, arg("a_epsilon")=1e-8), "Compares this JFABase with the 'other' one to be approximately the same.")
    .def("load", &bob::machine::JFAMachine::load, (arg("self"), arg("config")), "Loads the configuration parameters from a configuration file.")
    .def("save", &bob::machine::JFAMachine::save, (arg("self"), arg("config")), "Saves the configuration parameters to a configuration file.")
    .def("estimate_x", &py_jfa_estimateX, (arg("self"), arg("stats"), arg("x")), "Estimates the session offset x (LPT assumption) given GMM statistics.")
    .def("estimate_ux", &py_jfa_estimateUx, (arg("self"), arg("stats"), arg("ux")), "Estimates Ux (LPT assumption) given GMM statistics.")
    .def("forward_ux", &py_jfa_forwardUx, (arg("self"), arg("stats"), arg("ux")), "Processes the GMM statistics and Ux to return a score.")
    .add_property("jfa_base", &bob::machine::JFAMachine::getJFABase, &bob::machine::JFAMachine::setJFABase)
    .add_property("x", &py_jfa_getX)
    .add_property("y", &py_jfa_getY, &py_jfa_setY)
    .add_property("z", &py_jfa_getZ, &py_jfa_setZ)
    .add_property("dim_c", &bob::machine::JFAMachine::getDimC)
    .add_property("dim_d", &bob::machine::JFAMachine::getDimD)
    .add_property("dim_cd", &bob::machine::JFAMachine::getDimCD)
    .add_property("dim_ru", &bob::machine::JFAMachine::getDimRu)
    .add_property("dim_rv", &bob::machine::JFAMachine::getDimRv)
  ;

  class_<bob::machine::ISVBase, boost::shared_ptr<bob::machine::ISVBase>, bases<bob::machine::Machine<bob::machine::GMMStats, double> > >("ISVBase", "A ISVBase A ISVBase instance can be seen as a container for U and D when performing Joint Factor Analysis (ISV). TODO: add references", init<const boost::shared_ptr<bob::machine::GMMMachine>, optional<const size_t> >((arg("ubm"), arg("ru")=1), "Builds a new ISVBase."))
    .def(init<>("Constructs a 1 ISVBase instance. You have to set a UBM GMM and resize the U and D subspaces afterwards."))
    .def(init<bob::io::HDF5File&>((arg("config")), "Constructs a new ISVBaseMachine from a configuration file."))
    .def(init<const bob::machine::ISVBase&>((arg("machine")), "Copy constructs a ISVBase"))
    .def(self == self)
    .def(self != self)
    .def("is_similar_to", &bob::machine::ISVBase::is_similar_to, (arg("self"), arg("other"), arg("r_epsilon")=1e-5, arg("a_epsilon")=1e-8), "Compares this ISVBase with the 'other' one to be approximately the same.")
    .def("load", &bob::machine::ISVBase::load, (arg("self"), arg("config")), "Loads the configuration parameters from a configuration file.")
    .def("save", &bob::machine::ISVBase::save, (arg("self"), arg("config")), "Saves the configuration parameters to a configuration file.")
    .def("resize", &bob::machine::ISVBase::resize, (arg("self"), arg("ru")), "Reset the dimensionality of the subspaces U.")
    .add_property("ubm", &bob::machine::ISVBase::getUbm, &bob::machine::ISVBase::setUbm)
    .add_property("u", &py_isv_getU, &py_isv_setU)
    .add_property("d", &py_isv_getD, &py_isv_setD)
    .add_property("dim_c", &bob::machine::ISVBase::getDimC)
    .add_property("dim_d", &bob::machine::ISVBase::getDimD)
    .add_property("dim_cd", &bob::machine::ISVBase::getDimCD)
    .add_property("dim_ru", &bob::machine::ISVBase::getDimRu)
  ;

  class_<bob::machine::ISVMachine, boost::shared_ptr<bob::machine::ISVMachine>, bases<bob::machine::Machine<bob::machine::GMMStats, double> > >("ISVMachine", "A ISVMachine. An attached ISVBase should be provided for Joint Factor Analysis. The ISVMachine carries information about the speaker factors z, whereas a ISVBase carries information about the matrices U and D.", init<const boost::shared_ptr<bob::machine::ISVBase> >((arg("isv_base")), "Builds a new ISVMachine."))
    .def(init<>("Constructs a 1 ISVMachine instance. You have to set a ISVBase afterwards."))
    .def(init<bob::io::HDF5File&>((arg("config")), "Constructs a new ISVMachine from a configuration file."))
    .def(init<const bob::machine::ISVMachine&>((arg("machine")), "Copy constructs a ISVMachine"))
    .def(self == self)
    .def(self != self)
    .def("is_similar_to", &bob::machine::ISVMachine::is_similar_to, (arg("self"), arg("other"), arg("r_epsilon")=1e-5, arg("a_epsilon")=1e-8), "Compares this ISVBase with the 'other' one to be approximately the same.")
    .def("load", &bob::machine::ISVMachine::load, (arg("self"), arg("config")), "Loads the configuration parameters from a configuration file.")
    .def("save", &bob::machine::ISVMachine::save, (arg("self"), arg("config")), "Saves the configuration parameters to a configuration file.")
    .def("estimate_x", &py_isv_estimateX, (arg("self"), arg("stats"), arg("x")), "Estimates the session offset x (LPT assumption) given GMM statistics.")
    .def("estimate_ux", &py_isv_estimateUx, (arg("self"), arg("stats"), arg("ux")), "Estimates Ux (LPT assumption) given GMM statistics.")
    .def("forward_ux", &py_isv_forwardUx, (arg("self"), arg("stats"), arg("ux")), "Processes the GMM statistics and Ux to return a score.")
    .add_property("isv_base", &bob::machine::ISVMachine::getISVBase, &bob::machine::ISVMachine::setISVBase)
    .add_property("x", &py_isv_getX)
    .add_property("z", &py_isv_getZ, &py_isv_setZ)
    .add_property("dim_c", &bob::machine::ISVMachine::getDimC)
    .add_property("dim_d", &bob::machine::ISVMachine::getDimD)
    .add_property("dim_cd", &bob::machine::ISVMachine::getDimCD)
    .add_property("dim_ru", &bob::machine::ISVMachine::getDimRu)
  ;
}
