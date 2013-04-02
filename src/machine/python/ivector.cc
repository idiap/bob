/**
* @file machine/python/ivector.cc
* @date Sun Mar 31 18:07:00 2013 +0200
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
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <boost/python.hpp>
#include <bob/core/python/ndarray.h>
#include <boost/shared_ptr.hpp>
#include <bob/core/python/exception.h>
#include <bob/machine/IVectorMachine.h>

using namespace boost::python;

static object py_iv_getT(const bob::machine::IVectorMachine& machine)
{
  const blitz::Array<double,2> Tm = machine.getT();
  bob::python::ndarray T(bob::core::array::t_float64, Tm.extent(0), Tm.extent(1));
  blitz::Array<double,2> T_ = T.bz<double,2>();
  T_ = Tm;
  return T.self();
}

static void py_iv_setT(bob::machine::IVectorMachine& machine,
  bob::python::const_ndarray T)
{
  const blitz::Array<double,2> T_ = T.bz<double,2>();
  machine.setT(T_);
}

static object py_iv_getSigma(const bob::machine::IVectorMachine& machine)
{
  const blitz::Array<double,1> sigmam = machine.getSigma();
  bob::python::ndarray sigma(bob::core::array::t_float64, sigmam.extent(0));
  blitz::Array<double,1> sigma_ = sigma.bz<double,1>();
  sigma_ = sigmam;
  return sigma.self();
}

static void py_iv_setSigma(bob::machine::IVectorMachine& machine,
  bob::python::const_ndarray sigma)
{
  const blitz::Array<double,1> sigma_ = sigma.bz<double,1>();
  machine.setSigma(sigma_);
}

static void py_computeIdTtSigmaInvT1(const bob::machine::IVectorMachine& machine,
  const bob::machine::GMMStats& gs, bob::python::ndarray output)
{
  blitz::Array<double,2> output_ = output.bz<double,2>();
  machine.computeIdTtSigmaInvT(gs, output_);
}

static object py_computeIdTtSigmaInvT2(const bob::machine::IVectorMachine& machine,
  const bob::machine::GMMStats& gs)
{
  bob::python::ndarray output(bob::core::array::t_float64, machine.getDimRt(), machine.getDimRt());
  blitz::Array<double,2> output_ = output.bz<double,2>();
  machine.computeIdTtSigmaInvT(gs, output_);
  return output.self();
}

static void py_computeTtSigmaInvFnorm1(const bob::machine::IVectorMachine& machine,
  const bob::machine::GMMStats& gs, bob::python::ndarray output)
{
  blitz::Array<double,1> output_ = output.bz<double,1>();
  machine.computeTtSigmaInvFnorm(gs, output_);
}

static object py_computeTtSigmaInvFnorm2(const bob::machine::IVectorMachine& machine,
  const bob::machine::GMMStats& gs)
{
  bob::python::ndarray output(bob::core::array::t_float64, machine.getDimRt());
  blitz::Array<double,1> output_ = output.bz<double,1>();
  machine.computeTtSigmaInvFnorm(gs, output_);
  return output.self();
}

static void py_iv_forward1(const bob::machine::IVectorMachine& machine,
  const bob::machine::GMMStats& gs, bob::python::ndarray ivector)
{
  blitz::Array<double,1> ivector_ = ivector.bz<double,1>();
  machine.forward(gs, ivector_);
}

static void py_iv_forward1_(const bob::machine::IVectorMachine& machine,
  const bob::machine::GMMStats& gs, bob::python::ndarray ivector)
{
  blitz::Array<double,1> ivector_ = ivector.bz<double,1>();
  machine.forward_(gs, ivector_);
}

static object py_iv_forward2(const bob::machine::IVectorMachine& machine,
  const bob::machine::GMMStats& gs)
{
  bob::python::ndarray ivector(bob::core::array::t_float64, machine.getDimRt());
  blitz::Array<double,1> ivector_ = ivector.bz<double,1>();
  machine.forward(gs, ivector_);
  return ivector.self();
}


void bind_machine_ivector()
{
  // TODO: reuse binding from generic machine
  class_<bob::machine::IVectorMachine, boost::shared_ptr<bob::machine::IVectorMachine> >("IVectorMachine", "An IVectorMachine to extract i-vector (TODO: references)", init<boost::shared_ptr<bob::machine::GMMMachine>, optional<const size_t, const size_t> >((arg("ubm"), arg("rt")=1, arg("variance_threshold")=1e-10), "Builds a new IVectorMachine."))
    .def(init<>("Constructs a new empty IVectorMachine."))
    .def(init<bob::io::HDF5File&>((arg("config")), "Constructs a new IVectorMachine from a configuration file."))
    .def(init<const bob::machine::IVectorMachine&>((arg("machine")), "Copy constructs an IVectorMachine"))
    .def(self == self)
    .def(self != self)
    .def("is_similar_to", &bob::machine::IVectorMachine::is_similar_to, (arg("self"), arg("other"), arg("r_epsilon")=1e-5, arg("a_epsilon")=1e-8), "Compares this IVectorMachine with the 'other' one to be approximately the same.")
    .def("load", &bob::machine::IVectorMachine::load, (arg("self"), arg("config")), "Loads the configuration parameters from a configuration file.")
    .def("save", &bob::machine::IVectorMachine::save, (arg("self"), arg("config")), "Saves the configuration parameters to a configuration file.")
    .def("resize", &bob::machine::IVectorMachine::resize, (arg("self"), arg("rt")), "Reset the dimensionality of the Total Variability subspace T.")
    .add_property("ubm", &bob::machine::IVectorMachine::getUbm, &bob::machine::IVectorMachine::setUbm)
    .add_property("t", &py_iv_getT, &py_iv_setT)
    .add_property("sigma", &py_iv_getSigma, &py_iv_setSigma)
    .add_property("variance_threshold", &bob::machine::IVectorMachine::getVarianceThreshold, &bob::machine::IVectorMachine::setVarianceThreshold)
    .add_property("dim_c", &bob::machine::IVectorMachine::getDimC)
    .add_property("dim_d", &bob::machine::IVectorMachine::getDimD)
    .add_property("dim_cd", &bob::machine::IVectorMachine::getDimCD)
    .add_property("dim_rt", &bob::machine::IVectorMachine::getDimRt)
    .def("__compute_Id_TtSigmaInvT__", &py_computeIdTtSigmaInvT1, (arg("self"), arg("gmmstats"), arg("output")), "Computes (Id + sum_{c=1}^{C} N_{i,j,c} T^{T} Sigma_{c}^{-1} T)")
    .def("__compute_Id_TtSigmaInvT__", &py_computeIdTtSigmaInvT2, (arg("self"), arg("gmmstats")), "Computes (Id + sum_{c=1}^{C} N_{i,j,c} T^{T} Sigma_{c}^{-1} T)")
    .def("__compute_TtSigmaInvFnorm__", &py_computeTtSigmaInvFnorm1, (arg("self"), arg("gmmstats"), arg("output")), "Computes T^{T} Sigma^{-1} sum_{c=1}^{C} (F_c - N_c mean(c))")
    .def("__compute_TtSigmaInvFnorm__", &py_computeTtSigmaInvFnorm2, (arg("self"), arg("gmmstats")), "Computes T^{T} Sigma^{-1} sum_{c=1}^{C} (F_c - N_c mean(c))")
    .def("__call__", &py_iv_forward1_, (arg("self"), arg("gmmstats"), arg("ivector")), "Executes the machine on the GMMStats, and updates the ivector array. NO CHECK is performed.")
    .def("__call__", &py_iv_forward2, (arg("self"), arg("gmmstats")), "Executes the machine on the GMMStats. The ivector is allocated an returned.")
    .def("forward", &py_iv_forward1, (arg("self"), arg("gmmstats"), arg("ivector")), "Executes the machine on the GMMStats, and updates the ivector array.")
    .def("forward_", &py_iv_forward1_, (arg("self"), arg("gmmstats"), arg("ivector")), "Executes the machine on the GMMStats, and updates the ivector array. NO CHECK is performed.")
    .def("forward", &py_iv_forward2, (arg("self"), arg("gmmstats")), "Executes the machine on the GMMStats. The ivector is allocated an returned.")
  ;
}
