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
#include <bob/python/ndarray.h>
#include <boost/shared_ptr.hpp>
#include <bob/machine/JFAMachine.h>
#include <bob/machine/GMMMachine.h>

using namespace boost::python;

static void py_jfa_setU(bob::machine::JFABase& machine, 
  bob::python::const_ndarray U) 
{
  machine.setU(U.bz<double,2>());
}

static void py_jfa_setV(bob::machine::JFABase& machine,
  bob::python::const_ndarray V) 
{
  machine.setV(V.bz<double,2>());
}

static void py_jfa_setD(bob::machine::JFABase& machine,
  bob::python::const_ndarray D)
{
  machine.setD(D.bz<double,1>());
}

static void py_jfa_setY(bob::machine::JFAMachine& machine, bob::python::const_ndarray Y) {
  const blitz::Array<double,1>& Y_ = Y.bz<double,1>();
  machine.setY(Y_);
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
  double score;
  machine.forward(gmm_stats, ux.bz<double,1>(), score);
  return score;
}


static void py_isv_setU(bob::machine::ISVBase& machine, 
  bob::python::const_ndarray U) 
{
  machine.setU(U.bz<double,2>());
}

static void py_isv_setD(bob::machine::ISVBase& machine,
  bob::python::const_ndarray D)
{
  machine.setD(D.bz<double,1>());
}

static void py_isv_setZ(bob::machine::ISVMachine& machine, bob::python::const_ndarray Z) {
  machine.setZ(Z.bz<double,1>());
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
  double score;
  machine.forward(gmm_stats, ux.bz<double,1>(), score);
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


  class_<bob::machine::JFABase, boost::shared_ptr<bob::machine::JFABase>, bases<bob::machine::Machine<bob::machine::GMMStats, double> > >("JFABase", "A JFABase instance can be seen as a container for U, V and D when performing Joint Factor Analysis (JFA).\n\nReferences:\n[1] 'Explicit Modelling of Session Variability for Speaker Verification', R. Vogt, S. Sridharan, Computer Speech & Language, 2008, vol. 22, no. 1, pp. 17-38\n[2] 'Session Variability Modelling for Face Authentication', C. McCool, R. Wallace, M. McLaren, L. El Shafey, S. Marcel, IET Biometrics, 2013", init<const boost::shared_ptr<bob::machine::GMMMachine>, optional<const size_t, const size_t> >((arg("self"), arg("ubm"), arg("ru")=1, arg("rv")=1), "Builds a new JFABase."))
    .def(init<>((arg("self")), "Constructs a 1x1 JFABase instance. You have to set a UBM GMM and resize the U, V and D subspaces afterwards."))
    .def(init<bob::io::HDF5File&>((arg("self"), arg("config")), "Constructs a new JFABaseMachine from a configuration file."))
    .def(init<const bob::machine::JFABase&>((arg("self"), arg("machine")), "Copy constructs a JFABase"))
    .def(self == self)
    .def(self != self)
    .def("is_similar_to", &bob::machine::JFABase::is_similar_to, (arg("self"), arg("other"), arg("r_epsilon")=1e-5, arg("a_epsilon")=1e-8), "Compares this JFABase with the 'other' one to be approximately the same.")
    .def("load", &bob::machine::JFABase::load, (arg("self"), arg("config")), "Loads the configuration parameters from a configuration file.")
    .def("save", &bob::machine::JFABase::save, (arg("self"), arg("config")), "Saves the configuration parameters to a configuration file.")
    .def("resize", &bob::machine::JFABase::resize, (arg("self"), arg("ru"), arg("rv")), "Reset the dimensionality of the subspaces U and V.")
    .add_property("ubm", &bob::machine::JFABase::getUbm, &bob::machine::JFABase::setUbm, "The UBM GMM attached to this Joint Factor Analysis model")
    .add_property("u", make_function(&bob::machine::JFABase::getU, return_value_policy<copy_const_reference>()), &py_jfa_setU, "The subspace U for within-class variations")
    .add_property("v", make_function(&bob::machine::JFABase::getV, return_value_policy<copy_const_reference>()), &py_jfa_setV, "The subspace V for between-class variations")
    .add_property("d", make_function(&bob::machine::JFABase::getD, return_value_policy<copy_const_reference>()), &py_jfa_setD, "The subspace D for residual variations")
    .add_property("dim_c", &bob::machine::JFABase::getDimC, "The number of Gaussian components")
    .add_property("dim_d", &bob::machine::JFABase::getDimD, "The dimensionality of the feature space")
    .add_property("dim_cd", &bob::machine::JFABase::getDimCD, "The dimensionality of the supervector space")
    .add_property("dim_ru", &bob::machine::JFABase::getDimRu, "The dimensionality of the within-class variations subspace (rank of U)")
    .add_property("dim_rv", &bob::machine::JFABase::getDimRv, "The dimensionality of the between-class variations subspace (rank of V)")
  ;

  class_<bob::machine::JFAMachine, boost::shared_ptr<bob::machine::JFAMachine>, bases<bob::machine::Machine<bob::machine::GMMStats, double> > >("JFAMachine", "A JFAMachine. An attached JFABase should be provided for Joint Factor Analysis. The JFAMachine carries information about the speaker factors y and z, whereas a JFABase carries information about the matrices U, V and D.\n\nReferences:\n[1] 'Explicit Modelling of Session Variability for Speaker Verification', R. Vogt, S. Sridharan, Computer Speech & Language, 2008, vol. 22, no. 1, pp. 17-38\n[2] 'Session Variability Modelling for Face Authentication', C. McCool, R. Wallace, M. McLaren, L. El Shafey, S. Marcel, IET Biometrics, 2013", init<const boost::shared_ptr<bob::machine::JFABase> >((arg("self"), arg("jfa_base")), "Builds a new JFAMachine."))
    .def(init<>((arg("self")), "Constructs a 1x1 JFAMachine instance. You have to set a JFABase afterwards."))
    .def(init<bob::io::HDF5File&>((arg("self"), arg("config")), "Constructs a new JFAMachine from a configuration file."))
    .def(init<const bob::machine::JFAMachine&>((arg("self"), arg("machine")), "Copy constructs a JFAMachine"))
    .def(self == self)
    .def(self != self)
    .def("is_similar_to", &bob::machine::JFAMachine::is_similar_to, (arg("self"), arg("other"), arg("r_epsilon")=1e-5, arg("a_epsilon")=1e-8), "Compares this JFABase with the 'other' one to be approximately the same.")
    .def("load", &bob::machine::JFAMachine::load, (arg("self"), arg("config")), "Loads the configuration parameters from a configuration file.")
    .def("save", &bob::machine::JFAMachine::save, (arg("self"), arg("config")), "Saves the configuration parameters to a configuration file.")
    .def("estimate_x", &py_jfa_estimateX, (arg("self"), arg("stats"), arg("x")), "Estimates the session offset x (LPT assumption) given GMM statistics.")
    .def("estimate_ux", &py_jfa_estimateUx, (arg("self"), arg("stats"), arg("ux")), "Estimates Ux (LPT assumption) given GMM statistics.")
    .def("forward_ux", &py_jfa_forwardUx, (arg("self"), arg("stats"), arg("ux")), "Processes the GMM statistics and Ux to return a score.")
    .add_property("jfa_base", &bob::machine::JFAMachine::getJFABase, &bob::machine::JFAMachine::setJFABase, "The JFABase attached to this machine")
    .add_property("x", make_function(&bob::machine::JFAMachine::getX, return_value_policy<copy_const_reference>()), "The latent variable x (last one computed)")
    .add_property("y", make_function(&bob::machine::JFAMachine::getY, return_value_policy<copy_const_reference>()), &py_jfa_setY, "The latent variable y of this machine")
    .add_property("z", make_function(&bob::machine::JFAMachine::getZ, return_value_policy<copy_const_reference>()), &py_jfa_setZ, "The latent variable z of this machine")
    .add_property("dim_c", &bob::machine::JFAMachine::getDimC, "The number of Gaussian components")
    .add_property("dim_d", &bob::machine::JFAMachine::getDimD, "The dimensionality of the feature space")
    .add_property("dim_cd", &bob::machine::JFAMachine::getDimCD, "The dimensionality of the supervector space")
    .add_property("dim_ru", &bob::machine::JFAMachine::getDimRu, "The dimensionality of the within-class variations subspace (rank of U)")
    .add_property("dim_rv", &bob::machine::JFAMachine::getDimRv, "The dimensionality of the between-class variations subspace (rank of V)")
  ;

  class_<bob::machine::ISVBase, boost::shared_ptr<bob::machine::ISVBase>, bases<bob::machine::Machine<bob::machine::GMMStats, double> > >("ISVBase", "An ISVBase instance can be seen as a container for U and D when performing Joint Factor Analysis (ISV). \n\nReferences:\n[1] 'Explicit Modelling of Session Variability for Speaker Verification', R. Vogt, S. Sridharan, Computer Speech & Language, 2008, vol. 22, no. 1, pp. 17-38\n[2] 'Session Variability Modelling for Face Authentication', C. McCool, R. Wallace, M. McLaren, L. El Shafey, S. Marcel, IET Biometrics, 2013", init<const boost::shared_ptr<bob::machine::GMMMachine>, optional<const size_t> >((arg("self"), arg("ubm"), arg("ru")=1), "Builds a new ISVBase."))
    .def(init<>((arg("self")), "Constructs a 1 ISVBase instance. You have to set a UBM GMM and resize the U and D subspaces afterwards."))
    .def(init<bob::io::HDF5File&>((arg("self"), arg("config")), "Constructs a new ISVBaseMachine from a configuration file."))
    .def(init<const bob::machine::ISVBase&>((arg("self"), arg("machine")), "Copy constructs an ISVBase"))
    .def(self == self)
    .def(self != self)
    .def("is_similar_to", &bob::machine::ISVBase::is_similar_to, (arg("self"), arg("other"), arg("r_epsilon")=1e-5, arg("a_epsilon")=1e-8), "Compares this ISVBase with the 'other' one to be approximately the same.")
    .def("load", &bob::machine::ISVBase::load, (arg("self"), arg("config")), "Loads the configuration parameters from a configuration file.")
    .def("save", &bob::machine::ISVBase::save, (arg("self"), arg("config")), "Saves the configuration parameters to a configuration file.")
    .def("resize", &bob::machine::ISVBase::resize, (arg("self"), arg("ru")), "Reset the dimensionality of the subspaces U.")
    .add_property("ubm", &bob::machine::ISVBase::getUbm, &bob::machine::ISVBase::setUbm, "The UBM GMM attached to this Joint Factor Analysis model")
    .add_property("u", make_function(&bob::machine::ISVBase::getU, return_value_policy<copy_const_reference>()), &py_isv_setU, "The subspace U for within-class variations")
    .add_property("d", make_function(&bob::machine::ISVBase::getD, return_value_policy<copy_const_reference>()), &py_isv_setD, "The subspace D for residual variations")
    .add_property("dim_c", &bob::machine::ISVBase::getDimC, "The number of Gaussian components")
    .add_property("dim_d", &bob::machine::ISVBase::getDimD, "The dimensionality of the feature space")
    .add_property("dim_cd", &bob::machine::ISVBase::getDimCD, "The dimensionality of the supervector space")
    .add_property("dim_ru", &bob::machine::ISVBase::getDimRu, "The dimensionality of the within-class variations subspace (rank of U)")
  ;

  class_<bob::machine::ISVMachine, boost::shared_ptr<bob::machine::ISVMachine>, bases<bob::machine::Machine<bob::machine::GMMStats, double> > >("ISVMachine", "An ISVMachine. An attached ISVBase should be provided for Inter-session Variability Modelling. The ISVMachine carries information about the speaker factors z, whereas a ISVBase carries information about the matrices U and D. \n\nReferences:\n[1] 'Explicit Modelling of Session Variability for Speaker Verification', R. Vogt, S. Sridharan, Computer Speech & Language, 2008, vol. 22, no. 1, pp. 17-38\n[2] 'Session Variability Modelling for Face Authentication', C. McCool, R. Wallace, M. McLaren, L. El Shafey, S. Marcel, IET Biometrics, 2013", init<const boost::shared_ptr<bob::machine::ISVBase> >((arg("self"), arg("isv_base")), "Builds a new ISVMachine."))
    .def(init<>((arg("self")), "Constructs a 1 ISVMachine instance. You have to set a ISVBase afterwards."))
    .def(init<bob::io::HDF5File&>((arg("self"), arg("config")), "Constructs a new ISVMachine from a configuration file."))
    .def(init<const bob::machine::ISVMachine&>((arg("self"), arg("machine")), "Copy constructs an ISVMachine"))
    .def(self == self)
    .def(self != self)
    .def("is_similar_to", &bob::machine::ISVMachine::is_similar_to, (arg("self"), arg("other"), arg("r_epsilon")=1e-5, arg("a_epsilon")=1e-8), "Compares this ISVBase with the 'other' one to be approximately the same.")
    .def("load", &bob::machine::ISVMachine::load, (arg("self"), arg("config")), "Loads the configuration parameters from a configuration file.")
    .def("save", &bob::machine::ISVMachine::save, (arg("self"), arg("config")), "Saves the configuration parameters to a configuration file.")
    .def("estimate_x", &py_isv_estimateX, (arg("self"), arg("stats"), arg("x")), "Estimates the session offset x (LPT assumption) given GMM statistics.")
    .def("estimate_ux", &py_isv_estimateUx, (arg("self"), arg("stats"), arg("ux")), "Estimates Ux (LPT assumption) given GMM statistics.")
    .def("forward_ux", &py_isv_forwardUx, (arg("self"), arg("stats"), arg("ux")), "Processes the GMM statistics and Ux to return a score.")
    .add_property("isv_base", &bob::machine::ISVMachine::getISVBase, &bob::machine::ISVMachine::setISVBase, "The ISVBase attached to this machine")
    .add_property("x", make_function(&bob::machine::ISVMachine::getX, return_value_policy<copy_const_reference>()), "The latent variable x (last one computed)")
    .add_property("z", make_function(&bob::machine::ISVMachine::getZ, return_value_policy<copy_const_reference>()), &py_isv_setZ, "The latent variable z of this machine")
    .add_property("dim_c", &bob::machine::ISVMachine::getDimC, "The number of Gaussian components")
    .add_property("dim_d", &bob::machine::ISVMachine::getDimD, "The dimensionality of the feature space")
    .add_property("dim_cd", &bob::machine::ISVMachine::getDimCD, "The dimensionality of the supervector space")
    .add_property("dim_ru", &bob::machine::ISVMachine::getDimRu, "The dimensionality of the within-class variations subspace (rank of U)")
  ;
}
