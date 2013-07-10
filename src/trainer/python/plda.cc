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
#include <bob/python/ndarray.h>
#include <boost/python/stl_iterator.hpp>
#include <bob/machine/PLDAMachine.h>
#include <bob/trainer/PLDATrainer.h>

using namespace boost::python;

typedef bob::trainer::EMTrainer<bob::machine::PLDABase, std::vector<blitz::Array<double,2> > > EMTrainerPLDA;

static void plda_train(EMTrainerPLDA& t, bob::machine::PLDABase& m, object data)
{
  stl_input_iterator<bob::python::const_ndarray> dbegin(data), dend;
  std::vector<bob::python::const_ndarray> vdata(dbegin, dend);
  std::vector<blitz::Array<double,2> > vdata_ref;
  for(std::vector<bob::python::const_ndarray>::iterator it=vdata.begin(); 
      it!=vdata.end(); ++it)
    vdata_ref.push_back(it->bz<double,2>());
  // Calls the train function
  t.train(m, vdata_ref);
}

static void plda_initialize(EMTrainerPLDA& t, bob::machine::PLDABase& m, object data)
{
  stl_input_iterator<bob::python::const_ndarray> dbegin(data), dend;
  std::vector<bob::python::const_ndarray> vdata(dbegin, dend);
  std::vector<blitz::Array<double,2> > vdata_ref;
  for(std::vector<bob::python::const_ndarray>::iterator it=vdata.begin(); 
      it!=vdata.end(); ++it)
    vdata_ref.push_back(it->bz<double,2>());
  // Calls the initialization function
  t.initialize(m, vdata_ref);
}

static void plda_eStep(EMTrainerPLDA& t, bob::machine::PLDABase& m, object data)
{
  stl_input_iterator<bob::python::const_ndarray> dbegin(data), dend;
  std::vector<bob::python::const_ndarray> vdata(dbegin, dend);
  std::vector<blitz::Array<double,2> > vdata_ref;
  for(std::vector<bob::python::const_ndarray>::iterator it=vdata.begin(); 
      it!=vdata.end(); ++it)
    vdata_ref.push_back(it->bz<double,2>());
  // Calls the eStep function
  t.eStep(m, vdata_ref);
}

static void plda_mStep(EMTrainerPLDA& t, bob::machine::PLDABase& m, object data)
{
  stl_input_iterator<bob::python::const_ndarray> dbegin(data), dend;
  std::vector<bob::python::const_ndarray> vdata(dbegin, dend);
  std::vector<blitz::Array<double,2> > vdata_ref;
  for(std::vector<bob::python::const_ndarray>::iterator it=vdata.begin(); 
      it!=vdata.end(); ++it)
    vdata_ref.push_back(it->bz<double,2>());
  // Calls the mStep function
  t.mStep(m, vdata_ref);
}

static void plda_finalize(EMTrainerPLDA& t, bob::machine::PLDABase& m, object data)
{ 
  stl_input_iterator<bob::python::const_ndarray> dbegin(data), dend;
  std::vector<bob::python::const_ndarray> vdata(dbegin, dend);
  std::vector<blitz::Array<double,2> > vdata_ref;
  for(std::vector<bob::python::const_ndarray>::iterator it=vdata.begin(); 
      it!=vdata.end(); ++it)
    vdata_ref.push_back(it->bz<double,2>());
  // Calls the finalization function
  t.finalize(m, vdata_ref);
}

static object get_z_first_order(bob::trainer::PLDATrainer& m) {
  const std::vector<blitz::Array<double,2> >& v = m.getZFirstOrder();
  list retval;
  for (size_t k=0; k<v.size(); ++k) retval.append(v[k]); //copy
  return tuple(retval);
}

static object get_z_second_order(bob::trainer::PLDATrainer& m) {
  const std::vector<blitz::Array<double,3> >& v = m.getZSecondOrder();
  list retval;
  for (size_t k=0; k<v.size(); ++k) retval.append(v[k]); //copy
  return tuple(retval);
}


void bind_trainer_plda() 
{
  class_<EMTrainerPLDA, boost::noncopyable>("EMTrainerPLDA", "The base python class for all EM/PLDA-based trainers.", no_init)
    .add_property("max_iterations", &EMTrainerPLDA::getMaxIterations, &EMTrainerPLDA::setMaxIterations, "Max iterations")
    .add_property("rng", &EMTrainerPLDA::getRng, &EMTrainerPLDA::setRng, "The Mersenne Twister mt19937 random generator used for the initialization of subspaces/arrays before the EM loop.")
    .def("train", &plda_train, (arg("self"), arg("machine"), arg("data")), "Trains a PLDABase using data (mu, F, G and sigma are learnt).")
    .def("initialize", &plda_initialize, (arg("self"), arg("machine"), arg("data")), "This method is called before the EM algorithm")
    .def("finalize", &plda_finalize, (arg("self"), arg("machine"), arg("data")), "This method is called at the end of the EM algorithm")
    .def("e_step", &plda_eStep, (arg("self"), arg("machine"), arg("data")),
       "Updates the hidden variable distribution (or the sufficient statistics) given the Machine parameters. ")
    .def("m_step", &plda_mStep, (arg("self"), arg("machine"), arg("data")), "Updates the Machine parameters given the hidden variable distribution (or the sufficient statistics)")
  ;

  class_<bob::trainer::PLDATrainer, boost::noncopyable, bases<EMTrainerPLDA> > PLDAT("PLDATrainer", "A trainer for Probabilistic Linear Discriminant Analysis (PLDA). The train() method will learn the mu, F, G and Sigma of the model, whereas the enrol() method, will store model information about the enrolment samples for a specific class.\n\nReferences:\n1. 'A Scalable Formulation of Probabilistic Linear Discriminant Analysis: Applied to Face Recognition', Laurent El Shafey, Chris McCool, Roy Wallace, Sebastien Marcel, TPAMI'2013\n2. 'Probabilistic Linear Discriminant Analysis for Inference About Identity', Prince and Elder, ICCV'2007.\n3. 'Probabilistic Models for Inference about Identity', Li, Fu, Mohammed, Elder and Prince, TPAMI'2012.");
  
  PLDAT.def(init<optional<const size_t, const bool> >((arg("self"), arg("max_iterations")=100, arg("use_sum_second_order")=true),"Initializes a new PLDATrainer."))
    .def(init<const bob::trainer::PLDATrainer&>((arg("self"), arg("trainer")), "Copy constructs a PLDATrainer"))
    .def(self == self)
    .def(self != self)
    .def("is_similar_to", &bob::trainer::PLDATrainer::is_similar_to, (arg("self"), arg("other"), arg("r_epsilon")=1e-5, arg("a_epsilon")=1e-8), "Compares this PLDATrainer with the 'other' one to be approximately the same.")
    .def("enrol", &bob::trainer::PLDATrainer::enrol, (arg("self"), arg("plda_machine"), arg("data")), "Enrol a class-specific model (PLDAMachine) given a set of enrolment samples.")
    .add_property("use_sum_second_order", &bob::trainer::PLDATrainer::getUseSumSecondOrder, &bob::trainer::PLDATrainer::setUseSumSecondOrder, "Tells whether the second order statistics are stored during the training procedure, or only their sum.")
    .add_property("z_first_order", &get_z_first_order)
    .add_property("z_second_order", &get_z_second_order)
    .add_property("z_second_order_sum", make_function(&bob::trainer::PLDATrainer::getZSecondOrderSum, return_value_policy<copy_const_reference>()))
  ;

  // Sets the scope to the one of the PLDATrainer
  scope s(PLDAT);

  // Adds enums in the previously defined current scope
  enum_<bob::trainer::PLDATrainer::InitFMethod>("init_f_method")
    .value("RANDOM_F", bob::trainer::PLDATrainer::RANDOM_F)
    .value("BETWEEN_SCATTER", bob::trainer::PLDATrainer::BETWEEN_SCATTER)
    .export_values()
  ;   

  enum_<bob::trainer::PLDATrainer::InitGMethod>("init_g_method")
    .value("RANDOM_G", bob::trainer::PLDATrainer::RANDOM_G)
    .value("WITHIN_SCATTER", bob::trainer::PLDATrainer::WITHIN_SCATTER)
    .export_values()
  ;

  enum_<bob::trainer::PLDATrainer::InitSigmaMethod>("init_sigma_method")
    .value("RANDOM_SIGMA", bob::trainer::PLDATrainer::RANDOM_SIGMA)
    .value("VARIANCE_G", bob::trainer::PLDATrainer::VARIANCE_G)
    .value("CONSTANT", bob::trainer::PLDATrainer::CONSTANT)
    .value("VARIANCE_DATA", bob::trainer::PLDATrainer::VARIANCE_DATA)
    .export_values()
  ;

  // Binds randomization/enumration-related methods
  PLDAT.add_property("init_f_method", &bob::trainer::PLDATrainer::getInitFMethod, &bob::trainer::PLDATrainer::setInitFMethod, "The method used for the initialization of F.")
    .add_property("init_f_ratio", &bob::trainer::PLDATrainer::getInitFRatio, &bob::trainer::PLDATrainer::setInitFRatio, "The ratio used for the initialization of F.")
    .add_property("init_g_method", &bob::trainer::PLDATrainer::getInitGMethod, &bob::trainer::PLDATrainer::setInitGMethod, "The method used for the initialization of G.")
    .add_property("init_g_ratio", &bob::trainer::PLDATrainer::getInitGRatio, &bob::trainer::PLDATrainer::setInitGRatio, "The ratio used for the initialization of G.")
    .add_property("init_sigma_method", &bob::trainer::PLDATrainer::getInitSigmaMethod, &bob::trainer::PLDATrainer::setInitSigmaMethod, "The method used for the initialization of sigma.")
    .add_property("init_sigma_ratio", &bob::trainer::PLDATrainer::getInitSigmaRatio, &bob::trainer::PLDATrainer::setInitSigmaRatio, "The ratio used for the initialization of sigma.")
  ;
}
