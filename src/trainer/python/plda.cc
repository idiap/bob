/**
 * @file trainer/python/plda.cc
 * @date Fri Oct 14 18:07:56 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Python bindings to Probabilistic Linear Discriminant Analysis
 * trainers.
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
#include "bob/machine/PLDAMachine.h"
#include "bob/trainer/PLDATrainer.h"

using namespace boost::python;

static void plda_train(bob::trainer::PLDABaseTrainer& t, bob::machine::PLDABaseMachine& m, list l_arraysets)
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

static void plda_initialization(bob::trainer::PLDABaseTrainer& t, bob::machine::PLDABaseMachine& m, list l_arraysets)
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

static void plda_eStep(bob::trainer::PLDABaseTrainer& t, bob::machine::PLDABaseMachine& m, list l_arraysets)
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

static void plda_mStep(bob::trainer::PLDABaseTrainer& t, bob::machine::PLDABaseMachine& m, list l_arraysets)
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

static void plda_finalization(bob::trainer::PLDABaseTrainer& t, bob::machine::PLDABaseMachine& m, list l_arraysets)
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

static object get_z_first_order(bob::trainer::PLDABaseTrainer& m) {
  const std::vector<blitz::Array<double,2> >& v = m.getZFirstOrder();
  list retval;
  for (size_t k=0; k<v.size(); ++k) retval.append(v[k]); //copy
  return tuple(retval);
}

static object get_z_second_order(bob::trainer::PLDABaseTrainer& m) {
  const std::vector<blitz::Array<double,3> >& v = m.getZSecondOrder();
  list retval;
  for (size_t k=0; k<v.size(); ++k) retval.append(v[k]); //copy
  return tuple(retval);
}

void bind_trainer_plda() 
{
  typedef bob::trainer::EMTrainer<bob::machine::PLDABaseMachine, std::vector<blitz::Array<double,2> > > EMTrainerPLDABase; 

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

  enum_<bob::trainer::PLDABaseTrainer::InitFMethod>("init_f_method")
    .value("RANDOM_F", bob::trainer::PLDABaseTrainer::RANDOM_F)
    .value("BETWEEN_SCATTER", bob::trainer::PLDABaseTrainer::BETWEEN_SCATTER)
    ;   

  enum_<bob::trainer::PLDABaseTrainer::InitGMethod>("init_g_method")
    .value("RANDOM_G", bob::trainer::PLDABaseTrainer::RANDOM_G)
    .value("WITHIN_SCATTER", bob::trainer::PLDABaseTrainer::WITHIN_SCATTER)
    ;

  enum_<bob::trainer::PLDABaseTrainer::InitSigmaMethod>("init_sigma_method")
    .value("RANDOM_SIGMA", bob::trainer::PLDABaseTrainer::RANDOM_SIGMA)
    .value("VARIANCE_G", bob::trainer::PLDABaseTrainer::VARIANCE_G)
    .value("CONSTANT", bob::trainer::PLDABaseTrainer::CONSTANT)
    .value("VARIANCE_DATA", bob::trainer::PLDABaseTrainer::VARIANCE_DATA)
    ;

  class_<bob::trainer::PLDABaseTrainer, boost::noncopyable, bases<EMTrainerPLDABase> >("PLDABaseTrainer", "Creates a trainer for a PLDABaseMachine.", init<optional<double,double,bool,bool> >((arg("convergence_threshold")=0.001, arg("max_iterations")=10, arg("compute_likelihood")=false, arg("use_sum_second_order")=true),"Initializes a new PLDABaseTrainer."))
    .def(init<const bob::trainer::PLDABaseTrainer&>((arg("trainer")), "Copy constructs a PLDABaseTrainer"))
    .add_property("seed", &bob::trainer::PLDABaseTrainer::getSeed, &bob::trainer::PLDABaseTrainer::setSeed, "The seed used for the random initialization of F, G and sigma.")
    .add_property("use_sum_second_order", &bob::trainer::PLDABaseTrainer::getUseSumSecondOrder, &bob::trainer::PLDABaseTrainer::setUseSumSecondOrder, "Tells whether the second order statistics are stored during the training procedure, or only their sum.")
    .add_property("init_f_method", &bob::trainer::PLDABaseTrainer::getInitFMethod, &bob::trainer::PLDABaseTrainer::setInitFMethod, "The method used for the initialization of F.")
    .add_property("init_f_ratio", &bob::trainer::PLDABaseTrainer::getInitFRatio, &bob::trainer::PLDABaseTrainer::setInitFRatio, "The ratio used for the initialization of F.")
    .add_property("init_g_method", &bob::trainer::PLDABaseTrainer::getInitGMethod, &bob::trainer::PLDABaseTrainer::setInitGMethod, "The method used for the initialization of G.")
    .add_property("init_g_ratio", &bob::trainer::PLDABaseTrainer::getInitGRatio, &bob::trainer::PLDABaseTrainer::setInitGRatio, "The ratio used for the initialization of G.")
    .add_property("init_sigma_method", &bob::trainer::PLDABaseTrainer::getInitSigmaMethod, &bob::trainer::PLDABaseTrainer::setInitSigmaMethod, "The method used for the initialization of sigma.")
    .add_property("init_sigma_ratio", &bob::trainer::PLDABaseTrainer::getInitSigmaRatio, &bob::trainer::PLDABaseTrainer::setInitSigmaRatio, "The ratio used for the initialization of sigma.")
    .add_property("z_first_order", &get_z_first_order)
    .add_property("z_second_order", &get_z_second_order)
    .add_property("z_second_order_sum", make_function(&bob::trainer::PLDABaseTrainer::getZSecondOrderSum, return_value_policy<copy_const_reference>()))
    .def("train", &plda_train, (arg("self"), arg("machine"), arg("list_arraysets")), "Calls the training procedure. This will call initialization(), a loop of e_step() and m_step(), and finalization().")
    .def("initialization", &plda_initialization, (arg("self"), arg("machine"), arg("list_arraysets")), "Calls the initialization method of the training procedure.")
    .def("e_step", &plda_eStep, (arg("self"), arg("machine"), arg("list_arraysets")), "Calls the eStep method of the training procedure.")
    .def("m_step", &plda_mStep, (arg("self"), arg("machine"), arg("list_arraysets")), "Calls the mStep method of the training procedure.")
    .def("finalization", &plda_finalization, (arg("self"), arg("machine"), arg("list_arraysets")), "Calls the finalization method of the training procedure.")
    ;

  class_<bob::trainer::PLDATrainer, boost::noncopyable>("PLDATrainer", "Create a trainer for the PLDA.", init<>("Initializes a new PLDATrainer."))
    .def(init<const bob::trainer::PLDATrainer&>((arg("trainer")), "Copy constructs a PLDATrainer"))
    .def("enrol", &bob::trainer::PLDATrainer::enrol, (arg("self"), arg("plda_machine"), arg("arrayset")), "Call the enrollment procedure.")
    ;
}
