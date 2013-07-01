/**
* @file machine/python/ivector.cc
* @date Sun Mar 31 19:56:00 2013 +0200
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
#include <bob/python/ndarray.h>
#include <boost/shared_ptr.hpp>
#include <bob/trainer/IVectorTrainer.h>
#include <bob/machine/IVectorMachine.h>
#include <bob/trainer/EMTrainer.h>
#include <boost/python/stl_iterator.hpp>


using namespace boost::python;

static void py_train(bob::trainer::IVectorTrainer& trainer, 
  bob::machine::IVectorMachine& machine, object data)
{
  stl_input_iterator<bob::machine::GMMStats> dbegin(data), dend;
  std::vector<bob::machine::GMMStats> vdata(dbegin, dend);
  trainer.train(machine, vdata);
}

static void py_initialize(bob::trainer::IVectorTrainer& trainer, 
  bob::machine::IVectorMachine& machine, object data)
{
  stl_input_iterator<bob::machine::GMMStats> dbegin(data), dend;
  std::vector<bob::machine::GMMStats> vdata(dbegin, dend);
  trainer.initialize(machine, vdata);
}

static void py_eStep(bob::trainer::IVectorTrainer& trainer, 
  bob::machine::IVectorMachine& machine, object data)
{
  stl_input_iterator<bob::machine::GMMStats> dbegin(data), dend;
  std::vector<bob::machine::GMMStats> vdata(dbegin, dend);
  trainer.eStep(machine, vdata);
}

static void py_mStep(bob::trainer::IVectorTrainer& trainer, 
  bob::machine::IVectorMachine& machine, object data)
{
  stl_input_iterator<bob::machine::GMMStats> dbegin(data), dend;
  std::vector<bob::machine::GMMStats> vdata(dbegin, dend);
  trainer.mStep(machine, vdata);
}

static void py_finalize(bob::trainer::IVectorTrainer& trainer, 
  bob::machine::IVectorMachine& machine, object data)
{
  stl_input_iterator<bob::machine::GMMStats> dbegin(data), dend;
  std::vector<bob::machine::GMMStats> vdata(dbegin, dend);
  trainer.finalize(machine, vdata);
}

static object py_get_AccNijWij2(const bob::trainer::IVectorTrainer& trainer)
{
  const blitz::Array<double,3> acc_ref = trainer.getAccNijWij2();
  bob::python::ndarray acc(bob::core::array::t_float64, acc_ref.extent(0), acc_ref.extent(1), acc_ref.extent(2));
  blitz::Array<double,3> acc_ = acc.bz<double,3>();
  acc_ = acc_ref;
  return acc.self();
}

static void py_set_AccNijWij2(bob::trainer::IVectorTrainer& trainer,
  bob::python::const_ndarray acc)
{
  const blitz::Array<double,3> acc_ = acc.bz<double,3>();
  trainer.setAccNijWij2(acc_);
}

static object py_get_AccFnormijWij(const bob::trainer::IVectorTrainer& trainer)
{
  const blitz::Array<double,3> acc_ref = trainer.getAccFnormijWij();
  bob::python::ndarray acc(bob::core::array::t_float64, acc_ref.extent(0), acc_ref.extent(1), acc_ref.extent(2));
  blitz::Array<double,3> acc_ = acc.bz<double,3>();
  acc_ = acc_ref;
  return acc.self();
}

static void py_set_AccFnormijWij(bob::trainer::IVectorTrainer& trainer,
  bob::python::const_ndarray acc)
{
  const blitz::Array<double,3> acc_ = acc.bz<double,3>();
  trainer.setAccFnormijWij(acc_);
}

static object py_get_AccNij(const bob::trainer::IVectorTrainer& trainer)
{
  const blitz::Array<double,1> acc_ref = trainer.getAccNij();
  bob::python::ndarray acc(bob::core::array::t_float64, acc_ref.extent(0));
  blitz::Array<double,1> acc_ = acc.bz<double,1>();
  acc_ = acc_ref;
  return acc.self();
}

static void py_set_AccNij(bob::trainer::IVectorTrainer& trainer,
  bob::python::const_ndarray acc)
{
  const blitz::Array<double,1> acc_ = acc.bz<double,1>();
  trainer.setAccNij(acc_);
}

static object py_get_AccSnormij(const bob::trainer::IVectorTrainer& trainer)
{
  const blitz::Array<double,2> acc_ref = trainer.getAccSnormij();
  bob::python::ndarray acc(bob::core::array::t_float64, acc_ref.extent(0), acc_ref.extent(1));
  blitz::Array<double,2> acc_ = acc.bz<double,2>();
  acc_ = acc_ref;
  return acc.self();
}

static void py_set_AccSnormij(bob::trainer::IVectorTrainer& trainer,
  bob::python::const_ndarray acc)
{
  const blitz::Array<double,2> acc_ = acc.bz<double,2>();
  trainer.setAccSnormij(acc_);
}

void bind_trainer_ivector()
{
  typedef bob::trainer::EMTrainer<bob::machine::IVectorMachine, std::vector<bob::machine::GMMStats> > EMTrainerIVectorBase;

  class_<EMTrainerIVectorBase, boost::noncopyable>("EMTrainerIVector", "The base python class for all EM-based trainers.", no_init)
    .add_property("convergence_threshold", &EMTrainerIVectorBase::getConvergenceThreshold, &EMTrainerIVectorBase::setConvergenceThreshold, "Convergence threshold")
    .add_property("max_iterations", &EMTrainerIVectorBase::getMaxIterations, &EMTrainerIVectorBase::setMaxIterations, "Max iterations")
    .add_property("compute_likelihood_variable", &EMTrainerIVectorBase::getComputeLikelihood, &EMTrainerIVectorBase::setComputeLikelihood, "Indicates whether the log likelihood should be computed during EM or not")
    .add_property("rng", &EMTrainerIVectorBase::getRng, &EMTrainerIVectorBase::setRng, "The Mersenne Twister mt19937 random generator used for the initialization of subspaces/arrays before the EM loop.")
    .def("train", &EMTrainerIVectorBase::train, (arg("machine"), arg("data")), "Trains a machine using data")
    .def("initialize", &EMTrainerIVectorBase::initialize, (arg("machine"), arg("data")), "This method is called before the EM algorithm")
    .def("finalize", &EMTrainerIVectorBase::finalize, (arg("machine"), arg("data")), "This method is called at the end of the EM algorithm")
    .def("e_step", &EMTrainerIVectorBase::eStep, (arg("machine"), arg("data")),
       "Updates the hidden variable distribution (or the sufficient statistics) given the Machine parameters. ")
    .def("m_step", &EMTrainerIVectorBase::mStep, (arg("machine"), arg("data")), "Updates the Machine parameters given the hidden variable distribution (or the sufficient statistics)")
    .def("compute_likelihood", &EMTrainerIVectorBase::computeLikelihood, (arg("machine")), "Computes the current log likelihood given the hidden variable distribution (or the sufficient statistics)")
  ;


  class_<bob::trainer::IVectorTrainer, boost::shared_ptr<bob::trainer::IVectorTrainer>, boost::noncopyable, bases<EMTrainerIVectorBase> >("IVectorTrainer", "An IVectorTrainer to extract i-vector (TODO: references)", init<optional<bool, double, size_t, bool> >((arg("update_sigma")=false, arg("convergence_threshold")=0.001, arg("max_iterations")=10, arg("compute_likelihood")=false), "Builds a new IVectorTrainer."))
    .def(init<const bob::trainer::IVectorTrainer&>((arg("trainer")), "Copy constructs an IVectorTrainer"))
    .def(self == self)
    .def(self != self)
    .def("is_similar_to", &bob::trainer::IVectorTrainer::is_similar_to, (arg("self"), arg("other"), arg("r_epsilon")=1e-5, arg("a_epsilon")=1e-8), "Compares this IVectorTrainer with the 'other' one to be approximately the same.")
    .def("train", &py_train, (arg("self"), arg("machine"), arg("data")), "Trains a machine using data")
    .def("initialize", &py_initialize, (arg("self"), arg("machine"), arg("data")), "This method is called before the EM loop")
    .def("e_step", &py_eStep, (arg("self"), arg("machine"), arg("data")),
       "Updates the hidden variable distribution (or the sufficient statistics) given the Machine parameters. ")
    .def("m_step", &py_mStep, (arg("self"), arg("machine"), arg("data")), "Updates the Machine parameters given the hidden variable distribution (or the sufficient statistics)")
    .def("finalize", &py_finalize, (arg("self"), arg("machine"), arg("data")), "This method is called after the EM loop")
    .add_property("acc_nij_wij2", &py_get_AccNijWij2, &py_set_AccNijWij2, "Accumulator updated during the E-step")
    .add_property("acc_fnormij_wij", &py_get_AccFnormijWij, &py_set_AccFnormijWij, "Accumulator updated during the E-step")
    .add_property("acc_nij", &py_get_AccNij, &py_set_AccNij, "Accumulator updated during the E-step")
    .add_property("acc_snormij", &py_get_AccSnormij, &py_set_AccSnormij, "Accumulator updated during the E-step")
  ;
}
