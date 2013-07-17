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

typedef bob::trainer::EMTrainer<bob::machine::IVectorMachine, std::vector<bob::machine::GMMStats> > EMTrainerIVectorBase;

static void py_train(EMTrainerIVectorBase& trainer, 
  bob::machine::IVectorMachine& machine, object data)
{
  stl_input_iterator<bob::machine::GMMStats> dbegin(data), dend;
  std::vector<bob::machine::GMMStats> vdata(dbegin, dend);
  trainer.train(machine, vdata);
}

static void py_initialize(EMTrainerIVectorBase& trainer, 
  bob::machine::IVectorMachine& machine, object data)
{
  stl_input_iterator<bob::machine::GMMStats> dbegin(data), dend;
  std::vector<bob::machine::GMMStats> vdata(dbegin, dend);
  trainer.initialize(machine, vdata);
}

static void py_eStep(EMTrainerIVectorBase& trainer, 
  bob::machine::IVectorMachine& machine, object data)
{
  stl_input_iterator<bob::machine::GMMStats> dbegin(data), dend;
  std::vector<bob::machine::GMMStats> vdata(dbegin, dend);
  trainer.eStep(machine, vdata);
}

static void py_mStep(EMTrainerIVectorBase& trainer, 
  bob::machine::IVectorMachine& machine, object data)
{
  stl_input_iterator<bob::machine::GMMStats> dbegin(data), dend;
  std::vector<bob::machine::GMMStats> vdata(dbegin, dend);
  trainer.mStep(machine, vdata);
}

static void py_finalize(EMTrainerIVectorBase& trainer, 
  bob::machine::IVectorMachine& machine, object data)
{
  stl_input_iterator<bob::machine::GMMStats> dbegin(data), dend;
  std::vector<bob::machine::GMMStats> vdata(dbegin, dend);
  trainer.finalize(machine, vdata);
}

static void py_set_AccNijWij2(bob::trainer::IVectorTrainer& trainer,
  bob::python::const_ndarray acc)
{
  trainer.setAccNijWij2(acc.bz<double,3>());
}

static void py_set_AccFnormijWij(bob::trainer::IVectorTrainer& trainer,
  bob::python::const_ndarray acc)
{
  trainer.setAccFnormijWij(acc.bz<double,3>());
}

static void py_set_AccNij(bob::trainer::IVectorTrainer& trainer,
  bob::python::const_ndarray acc)
{
  trainer.setAccNij(acc.bz<double,1>());
}

static void py_set_AccSnormij(bob::trainer::IVectorTrainer& trainer,
  bob::python::const_ndarray acc)
{
  trainer.setAccSnormij(acc.bz<double,2>());
}

void bind_trainer_ivector()
{
  class_<EMTrainerIVectorBase, boost::noncopyable>("EMTrainerIVector", "The base python class for all EM-based trainers.", no_init)
    .add_property("convergence_threshold", &EMTrainerIVectorBase::getConvergenceThreshold, &EMTrainerIVectorBase::setConvergenceThreshold, "Convergence threshold")
    .add_property("max_iterations", &EMTrainerIVectorBase::getMaxIterations, &EMTrainerIVectorBase::setMaxIterations, "Max iterations")
    .add_property("compute_likelihood_variable", &EMTrainerIVectorBase::getComputeLikelihood, &EMTrainerIVectorBase::setComputeLikelihood, "Indicates whether the log likelihood should be computed during EM or not")
    .add_property("rng", &EMTrainerIVectorBase::getRng, &EMTrainerIVectorBase::setRng, "The Mersenne Twister mt19937 random generator used for the initialization of subspaces/arrays before the EM loop.")
    .def("train", &py_train, (arg("machine"), arg("data")), "Trains a machine using data")
    .def("initialize", &py_initialize, (arg("machine"), arg("data")), "This method is called before the EM algorithm")
    .def("finalize", &py_finalize, (arg("machine"), arg("data")), "This method is called at the end of the EM algorithm")
    .def("e_step", &py_eStep, (arg("machine"), arg("data")),
       "Updates the hidden variable distribution (or the sufficient statistics) given the Machine parameters. ")
    .def("m_step", &py_mStep, (arg("machine"), arg("data")), "Updates the Machine parameters given the hidden variable distribution (or the sufficient statistics)")
    .def("compute_likelihood", &EMTrainerIVectorBase::computeLikelihood, (arg("machine")), "Computes the current log likelihood given the hidden variable distribution (or the sufficient statistics)")
  ;


  class_<bob::trainer::IVectorTrainer, boost::shared_ptr<bob::trainer::IVectorTrainer>, boost::noncopyable, bases<EMTrainerIVectorBase> >("IVectorTrainer", "An trainer to extract i-vector (i.e. for training the Total Variability matrix)\n\nReferences:\n[1] 'Front End Factor Analysis for Speaker Verification', N. Dehak, P. Kenny, R. Dehak, P. Dumouchel, P. Ouellet, IEEE Transactions on Audio, Speech and Language Processing, 2010, vol. 19, issue 4, pp. 788-798", init<optional<bool, double, size_t, bool> >((arg("self"), arg("update_sigma")=false, arg("convergence_threshold")=0.001, arg("max_iterations")=10, arg("compute_likelihood")=false), "Builds a new IVectorTrainer."))
    .def(init<const bob::trainer::IVectorTrainer&>((arg("self"), arg("trainer")), "Copy constructs an IVectorTrainer"))
    .def(self == self)
    .def(self != self)
    .def("is_similar_to", &bob::trainer::IVectorTrainer::is_similar_to, (arg("self"), arg("other"), arg("r_epsilon")=1e-5, arg("a_epsilon")=1e-8), "Compares this IVectorTrainer with the 'other' one to be approximately the same.")
    .add_property("acc_nij_wij2", make_function(&bob::trainer::IVectorTrainer::getAccNijWij2, return_value_policy<copy_const_reference>()), &py_set_AccNijWij2, "Accumulator updated during the E-step")
    .add_property("acc_fnormij_wij", make_function(&bob::trainer::IVectorTrainer::getAccFnormijWij, return_value_policy<copy_const_reference>()), &py_set_AccFnormijWij, "Accumulator updated during the E-step")
    .add_property("acc_nij", make_function(&bob::trainer::IVectorTrainer::getAccNij, return_value_policy<copy_const_reference>()), &py_set_AccNij, "Accumulator updated during the E-step")
    .add_property("acc_snormij", make_function(&bob::trainer::IVectorTrainer::getAccSnormij, return_value_policy<copy_const_reference>()), &py_set_AccSnormij, "Accumulator updated during the E-step")
  ;
}
