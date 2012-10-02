/**
 * @file trainer/python/plda.cc
 * @date Fri Oct 14 18:07:56 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Python bindings to Probabilistic Linear Discriminant Analysis
 * trainers.
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
#include "bob/machine/PLDAMachine.h"
#include "bob/trainer/PLDATrainer.h"

using namespace boost::python;
namespace train = bob::trainer;
namespace mach = bob::machine;
namespace io = bob::io;


static void plda_train(train::PLDABaseTrainer& t, mach::PLDABaseMachine& m, list l_arraysets)
{
  int n_ids = len(l_arraysets);
  std::vector<blitz::Array<double,2> > v_arraysets;

  // Extracts the vector of Arraysets from the python list of Arraysets
  for(int id=0; id<n_ids; ++id) {
    blitz::Array<double,2> ar = extract<blitz::Array<double,2> >(l_arraysets[id]);
    v_arraysets.push_back(ar);
  }

  // Calls the train function
  t.train(m, v_arraysets);
}

static void plda_initialization(train::PLDABaseTrainer& t, mach::PLDABaseMachine& m, list l_arraysets)
{
  int n_ids = len(l_arraysets);
  std::vector<blitz::Array<double,2> > v_arraysets;

  // Extracts the vector of Arraysets from the python list of Arraysets
  for(int id=0; id<n_ids; ++id) {
    blitz::Array<double,2> ar = extract<blitz::Array<double,2> >(l_arraysets[id]);
    v_arraysets.push_back(ar);
  }

  // Calls the initialization function
  t.initialization(m, v_arraysets);
}

static void plda_eStep(train::PLDABaseTrainer& t, mach::PLDABaseMachine& m, list l_arraysets)
{
  int n_ids = len(l_arraysets);
  std::vector<blitz::Array<double,2> > v_arraysets;

  // Extracts the vector of Arraysets from the python list of Arraysets
  for(int id=0; id<n_ids; ++id) {
    blitz::Array<double,2> ar = extract<blitz::Array<double,2> >(l_arraysets[id]);
    v_arraysets.push_back(ar);
  }

  // Calls the eStep function
  t.eStep(m, v_arraysets);
}

static void plda_mStep(train::PLDABaseTrainer& t, mach::PLDABaseMachine& m, list l_arraysets)
{
  int n_ids = len(l_arraysets);
  std::vector<blitz::Array<double,2> > v_arraysets;

  // Extracts the vector of Arraysets from the python list of Arraysets
  for(int id=0; id<n_ids; ++id) {
    blitz::Array<double,2> ar = extract<blitz::Array<double,2> >(l_arraysets[id]);
    v_arraysets.push_back(ar);
  }

  // Calls the mStep function
  t.mStep(m, v_arraysets);
}

static void plda_finalization(train::PLDABaseTrainer& t, mach::PLDABaseMachine& m, list l_arraysets)
{
  int n_ids = len(l_arraysets);
  std::vector<blitz::Array<double,2> > v_arraysets;

  // Extracts the vector of Arraysets from the python list of Arraysets
  for(int id=0; id<n_ids; ++id) {
    blitz::Array<double,2> ar = extract<blitz::Array<double,2> >(l_arraysets[id]);
    v_arraysets.push_back(ar);
  }

  // Calls the finalization function
  t.finalization(m, v_arraysets);
}

static object get_z_first_order(train::PLDABaseTrainer& m) {
  const std::vector<blitz::Array<double,2> >& v = m.getZFirstOrder();
  list retval;
  for (size_t k=0; k<v.size(); ++k) retval.append(v[k]); //copy
  return tuple(retval);
}

void bind_trainer_plda() {
  typedef train::EMTrainer<mach::PLDABaseMachine, std::vector<blitz::Array<double,2> > > EMTrainerPLDABase; 

  class_<EMTrainerPLDABase, boost::noncopyable>("EMTrainerPLDA", "The base python class for all EM/PLDA-based trainers.", no_init)
    .add_property("convergence_threshold", &EMTrainerPLDABase::getConvergenceThreshold, &EMTrainerPLDABase::setConvergenceThreshold, "Convergence threshold")
    .add_property("max_iterations", &EMTrainerPLDABase::getMaxIterations, &EMTrainerPLDABase::setMaxIterations, "Max iterations")
    .add_property("compute_likelihood_variable", &EMTrainerPLDABase::getComputeLikelihood, &EMTrainerPLDABase::setComputeLikelihood, "Indicates whether the log likelihood should be computed during EM or not")
    .def("train", &EMTrainerPLDABase::train, (arg("machine"), arg("data")), "Trains a machine using data")
    .def("initialization", &EMTrainerPLDABase::initialization, (arg("machine"), arg("data")), "This method is called before the EM algorithm")
    .def("finalization", &EMTrainerPLDABase::finalization, (arg("machine"), arg("data")), "This method is called at the end of the EM algorithm")
    .def("e_step", &EMTrainerPLDABase::eStep, (arg("machine"), arg("data")),
       "Updates the hidden variable distribution (or the sufficient statistics) given the Machine parameters. ")
    .def("m_step", &EMTrainerPLDABase::mStep, (arg("machine"), arg("data")), "Updates the Machine parameters given the hidden variable distribution (or the sufficient statistics)")
    .def("compute_likelihood", &EMTrainerPLDABase::computeLikelihood, (arg("machine"), arg("data")), "Computes the current log likelihood given the hidden variable distribution (or the sufficient statistics)")
  ;


  class_<train::PLDABaseTrainer, boost::noncopyable, bases<EMTrainerPLDABase> >("PLDABaseTrainer", "Creates a trainer for a PLDABaseMachine.", init<int, int, optional<double,double,bool> >((arg("nf"), arg("ng"), arg("convergence_threshold"), arg("max_iterations"), arg("compute_likelihood")),"Initializes a new PLDABaseTrainer."))
    .add_property("seed", &train::PLDABaseTrainer::getSeed, &train::PLDABaseTrainer::setSeed, "The seed used for the random initialization of F, G and sigma.")
    .add_property("init_f_method", &train::PLDABaseTrainer::getInitFMethod, &train::PLDABaseTrainer::setInitFMethod, "The method used for the initialization of F.")
    .add_property("init_f_ratio", &train::PLDABaseTrainer::getInitFRatio, &train::PLDABaseTrainer::setInitFRatio, "The ratio used for the initialization of F.")
    .add_property("init_g_method", &train::PLDABaseTrainer::getInitGMethod, &train::PLDABaseTrainer::setInitGMethod, "The method used for the initialization of G.")
    .add_property("init_g_ratio", &train::PLDABaseTrainer::getInitGRatio, &train::PLDABaseTrainer::setInitGRatio, "The ratio used for the initialization of G.")
    .add_property("init_sigma_method", &train::PLDABaseTrainer::getInitSigmaMethod, &train::PLDABaseTrainer::setInitSigmaMethod, "The method used for the initialization of sigma.")
    .add_property("init_sigma_ratio", &train::PLDABaseTrainer::getInitSigmaRatio, &train::PLDABaseTrainer::setInitSigmaRatio, "The ratio used for the initialization of sigma.")
    .add_property("z_first_order", &get_z_first_order)
    .add_property("z_second_order_sum", make_function(&train::PLDABaseTrainer::getZSecondOrderSum, return_value_policy<copy_const_reference>()))
    .def("train", &plda_train, (arg("self"), arg("machine"), arg("list_arraysets")), "Calls the training procedure. This will call initialization(), a loop of e_step() and m_step(), and finalization().")
    .def("initialization", &plda_initialization, (arg("self"), arg("machine"), arg("list_arraysets")), "Calls the initialization method of the training procedure.")
    .def("e_step", &plda_eStep, (arg("self"), arg("machine"), arg("list_arraysets")), "Calls the eStep method of the training procedure.")
    .def("m_step", &plda_mStep, (arg("self"), arg("machine"), arg("list_arraysets")), "Calls the mStep method of the training procedure.")
    .def("finalization", &plda_finalization, (arg("self"), arg("machine"), arg("list_arraysets")), "Calls the finalization method of the training procedure.")
    ;


  class_<train::PLDATrainer, boost::noncopyable>("PLDATrainer", "Create a trainer for the PLDA.", init<mach::PLDAMachine&>((arg("plda")),"Initializes a new PLDATrainer."))
    .def("enrol", (void (train::PLDATrainer::*)(const blitz::Array<double,2>&))&train::PLDATrainer::enrol, (arg("self"), arg("arrayset")), "Call the enrollment procedure.")
    ;


}
