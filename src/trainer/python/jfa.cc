/**
 * @file trainer/python/jfa.cc
 * @date Tue Jul 19 12:16:17 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Python bindings to Joint Factor Analysis trainers
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

#include <bob/python/ndarray.h>
#include <boost/python/stl_iterator.hpp>
#include <bob/trainer/JFATrainer.h>
#include <boost/shared_ptr.hpp>

using namespace boost::python;


static object vector_as_list(const std::vector<blitz::Array<double,1> >& vec) 
{
  list retval;
  for(size_t k=0; k<vec.size(); ++k) 
  {
    const blitz::Array<double,1>& array = vec[k];
    bob::python::ndarray a(bob::core::array::t_float64, array.extent(0));
    blitz::Array<double,1> a_ = a.bz<double,1>();
    a_ = array;
    retval.append(a); //copy
  }
  return retval;
}

static object vector_as_list(const std::vector<blitz::Array<double,2> >& vec) 
{
  list retval;
  for(size_t k=0; k<vec.size(); ++k) 
  {
    const blitz::Array<double,2>& array = vec[k];
    bob::python::ndarray a(bob::core::array::t_float64, array.extent(0), array.extent(1));
    blitz::Array<double,2> a_ = a.bz<double,2>();
    a_ = array;
    retval.append(a); //copy
  }
  return retval;
}


static void extract_GMMStats(object data, 
  std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& training_data)
{
  stl_input_iterator<object> dbegin(data), dend;
  std::vector<object> vvdata(dbegin, dend);
  for (size_t i=0; i<vvdata.size(); ++i)
  {
    stl_input_iterator<boost::shared_ptr<bob::machine::GMMStats> > dlbegin(vvdata[i]), dlend;
    training_data.push_back(std::vector<boost::shared_ptr<bob::machine::GMMStats> >(dlbegin, dlend));
  }
}

static void isv_train(bob::trainer::ISVTrainer& t, bob::machine::ISVBase& m, object data)
{
  std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > > training_data;
  extract_GMMStats(data, training_data);
  // Calls the train function
  t.train(m, training_data);
}

static void isv_initialize(bob::trainer::ISVTrainer& t, bob::machine::ISVBase& m, object data)
{
  std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > > training_data;
  extract_GMMStats(data, training_data);
  // Calls the initialize function
  t.initialize(m, training_data);
}

static void isv_estep(bob::trainer::ISVTrainer& t, bob::machine::ISVBase& m, object data)
{
  std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > > training_data;
  extract_GMMStats(data, training_data);
  // Calls the E-Step function
  t.eStep(m, training_data);
}

static void isv_mstep(bob::trainer::ISVTrainer& t, bob::machine::ISVBase& m, object data)
{
  std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > > training_data;
  extract_GMMStats(data, training_data);
  // Calls the M-Step function
  t.mStep(m, training_data);
}

static void isv_finalize(bob::trainer::ISVTrainer& t, bob::machine::ISVBase& m, object data)
{
  std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > > training_data;
  extract_GMMStats(data, training_data);
  // Calls the finalization function
  t.finalize(m, training_data);
}

static void isv_enrol(bob::trainer::ISVTrainer& t, bob::machine::ISVMachine& m, object data, const size_t n_iter)
{
  stl_input_iterator<boost::shared_ptr<bob::machine::GMMStats> > dlbegin(data), dlend;
  std::vector<boost::shared_ptr<bob::machine::GMMStats> > vdata(dlbegin, dlend);
  // Calls the enrol function
  t.enrol(m, vdata, n_iter);
}

static object isv_get_x(const bob::trainer::ISVTrainer& t)
{
  return vector_as_list(t.getX());
}

static object isv_get_z(const bob::trainer::ISVTrainer& t)
{
  return vector_as_list(t.getZ());
}

static void isv_set_x(bob::trainer::ISVTrainer& t, object data)
{
  stl_input_iterator<bob::python::const_ndarray> vdata(data), dend;
  std::vector<blitz::Array<double,2> > vdata_ref;
  vdata_ref.reserve(len(data));
  for (; vdata != dend; ++vdata) vdata_ref.push_back((*vdata).bz<double,2>());
  t.setX(vdata_ref);
}

static void isv_set_z(bob::trainer::ISVTrainer& t, object data)
{
  stl_input_iterator<bob::python::const_ndarray> vdata(data), dend;
  std::vector<blitz::Array<double,1> > vdata_ref;
  vdata_ref.reserve(len(data));
  for (; vdata != dend; ++vdata) vdata_ref.push_back((*vdata).bz<double,1>());
  t.setZ(vdata_ref);
}


static object isv_get_accUA1(const bob::trainer::ISVTrainer& trainer)
{
  const blitz::Array<double,3>& acc_ref = trainer.getAccUA1();
  bob::python::ndarray acc(bob::core::array::t_float64, acc_ref.extent(0), acc_ref.extent(1), acc_ref.extent(2));
  blitz::Array<double,3> acc_ = acc.bz<double,3>();
  acc_ = acc_ref;
  return acc.self();
}

static void isv_set_accUA1(bob::trainer::ISVTrainer& trainer, 
  bob::python::const_ndarray acc)
{
  const blitz::Array<double,3> acc_ = acc.bz<double,3>();
  trainer.setAccUA1(acc_);
}

static object isv_get_accUA2(const bob::trainer::ISVTrainer& trainer)
{
  const blitz::Array<double,2>& acc_ref = trainer.getAccUA2();
  bob::python::ndarray acc(bob::core::array::t_float64, acc_ref.extent(0), acc_ref.extent(1));
  blitz::Array<double,2> acc_ = acc.bz<double,2>();
  acc_ = acc_ref;
  return acc.self();
}

static void isv_set_accUA2(bob::trainer::ISVTrainer& trainer, 
  bob::python::const_ndarray acc)
{
  const blitz::Array<double,2> acc_ = acc.bz<double,2>();
  trainer.setAccUA2(acc_);
}



static void jfa_train(bob::trainer::JFATrainer& t, bob::machine::JFABase& m, object data)
{
  std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > > training_data;
  extract_GMMStats(data, training_data);
  // Calls the train function
  t.train(m, training_data);
}

static void jfa_initialize(bob::trainer::JFATrainer& t, bob::machine::JFABase& m, object data)
{
  std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > > training_data;
  extract_GMMStats(data, training_data);
  // Calls the initialize function
  t.initialize(m, training_data);
}

static void jfa_estep1(bob::trainer::JFATrainer& t, bob::machine::JFABase& m, object data)
{
  std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > > training_data;
  extract_GMMStats(data, training_data);
  // Calls the E-Step function
  t.eStep1(m, training_data);
}

static void jfa_mstep1(bob::trainer::JFATrainer& t, bob::machine::JFABase& m, object data)
{
  std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > > training_data;
  extract_GMMStats(data, training_data);
  // Calls the M-Step function
  t.mStep1(m, training_data);
}

static void jfa_finalize1(bob::trainer::JFATrainer& t, bob::machine::JFABase& m, object data)
{
  std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > > training_data;
  extract_GMMStats(data, training_data);
  // Calls the finalization function
  t.finalize1(m, training_data);
}

static void jfa_estep2(bob::trainer::JFATrainer& t, bob::machine::JFABase& m, object data)
{
  std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > > training_data;
  extract_GMMStats(data, training_data);
  // Calls the E-Step function
  t.eStep2(m, training_data);
}

static void jfa_mstep2(bob::trainer::JFATrainer& t, bob::machine::JFABase& m, object data)
{
  std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > > training_data;
  extract_GMMStats(data, training_data);
  // Calls the M-Step function
  t.mStep2(m, training_data);
}

static void jfa_finalize2(bob::trainer::JFATrainer& t, bob::machine::JFABase& m, object data)
{
  std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > > training_data;
  extract_GMMStats(data, training_data);
  // Calls the finalization function
  t.finalize2(m, training_data);
}

static void jfa_estep3(bob::trainer::JFATrainer& t, bob::machine::JFABase& m, object data)
{
  std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > > training_data;
  extract_GMMStats(data, training_data);
  // Calls the E-Step function
  t.eStep3(m, training_data);
}

static void jfa_mstep3(bob::trainer::JFATrainer& t, bob::machine::JFABase& m, object data)
{
  std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > > training_data;
  extract_GMMStats(data, training_data);
  // Calls the M-Step function
  t.mStep3(m, training_data);
}

static void jfa_finalize3(bob::trainer::JFATrainer& t, bob::machine::JFABase& m, object data)
{
  std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > > training_data;
  extract_GMMStats(data, training_data);
  // Calls the finalization function
  t.finalize3(m, training_data);
}

static void jfa_train_loop(bob::trainer::JFATrainer& t, bob::machine::JFABase& m, object data)
{
  std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > > training_data;
  extract_GMMStats(data, training_data);
  // Calls the main loop function
  t.train_loop(m, training_data);
}

static void jfa_enrol(bob::trainer::JFATrainer& t, bob::machine::JFAMachine& m, object data, const size_t n_iter)
{
  stl_input_iterator<boost::shared_ptr<bob::machine::GMMStats> > dlbegin(data), dlend;
  std::vector<boost::shared_ptr<bob::machine::GMMStats> > vdata(dlbegin, dlend);
  // Calls the enrol function
  t.enrol(m, vdata, n_iter);
}

static object jfa_get_x(const bob::trainer::JFATrainer& t)
{
  return vector_as_list(t.getX());
}

static object jfa_get_y(const bob::trainer::JFATrainer& t)
{
  return vector_as_list(t.getY());
}

static object jfa_get_z(const bob::trainer::JFATrainer& t)
{
  return vector_as_list(t.getZ());
}

static void jfa_set_x(bob::trainer::JFATrainer& t, object data)
{
  stl_input_iterator<bob::python::const_ndarray> vdata(data), dend;
  std::vector<blitz::Array<double,2> > vdata_ref;
  vdata_ref.reserve(len(data));
  for (; vdata != dend; ++vdata) vdata_ref.push_back((*vdata).bz<double,2>());
  t.setX(vdata_ref);
}

static void jfa_set_y(bob::trainer::JFATrainer& t, object data)
{
  stl_input_iterator<bob::python::const_ndarray> vdata(data), dend;
  std::vector<blitz::Array<double,1> > vdata_ref;
  vdata_ref.reserve(len(data));
  for (; vdata != dend; ++vdata) vdata_ref.push_back((*vdata).bz<double,1>());
  t.setY(vdata_ref);
}

static void jfa_set_z(bob::trainer::JFATrainer& t, object data)
{
  stl_input_iterator<bob::python::const_ndarray> vdata(data), dend;
  std::vector<blitz::Array<double,1> > vdata_ref;
  vdata_ref.reserve(len(data));
  for (; vdata != dend; ++vdata) vdata_ref.push_back((*vdata).bz<double,1>());
  t.setZ(vdata_ref);
}


static object jfa_get_accUA1(const bob::trainer::JFATrainer& trainer)
{
  const blitz::Array<double,3>& acc_ref = trainer.getAccUA1();
  bob::python::ndarray acc(bob::core::array::t_float64, acc_ref.extent(0), acc_ref.extent(1), acc_ref.extent(2));
  blitz::Array<double,3> acc_ = acc.bz<double,3>();
  acc_ = acc_ref;
  return acc.self();
}

static void jfa_set_accUA1(bob::trainer::JFATrainer& trainer, 
  bob::python::const_ndarray acc)
{
  const blitz::Array<double,3> acc_ = acc.bz<double,3>();
  trainer.setAccUA1(acc_);
}

static object jfa_get_accUA2(const bob::trainer::JFATrainer& trainer)
{
  const blitz::Array<double,2>& acc_ref = trainer.getAccUA2();
  bob::python::ndarray acc(bob::core::array::t_float64, acc_ref.extent(0), acc_ref.extent(1));
  blitz::Array<double,2> acc_ = acc.bz<double,2>();
  acc_ = acc_ref;
  return acc.self();
}

static void jfa_set_accUA2(bob::trainer::JFATrainer& trainer, 
  bob::python::const_ndarray acc)
{
  const blitz::Array<double,2> acc_ = acc.bz<double,2>();
  trainer.setAccUA2(acc_);
}

static object jfa_get_accVA1(const bob::trainer::JFATrainer& trainer)
{
  const blitz::Array<double,3>& acc_ref = trainer.getAccVA1();
  bob::python::ndarray acc(bob::core::array::t_float64, acc_ref.extent(0), acc_ref.extent(1), acc_ref.extent(2));
  blitz::Array<double,3> acc_ = acc.bz<double,3>();
  acc_ = acc_ref;
  return acc.self();
}

static void jfa_set_accVA1(bob::trainer::JFATrainer& trainer, 
  bob::python::const_ndarray acc)
{
  const blitz::Array<double,3> acc_ = acc.bz<double,3>();
  trainer.setAccVA1(acc_);
}

static object jfa_get_accVA2(const bob::trainer::JFATrainer& trainer)
{
  const blitz::Array<double,2>& acc_ref = trainer.getAccVA2();
  bob::python::ndarray acc(bob::core::array::t_float64, acc_ref.extent(0), acc_ref.extent(1));
  blitz::Array<double,2> acc_ = acc.bz<double,2>();
  acc_ = acc_ref;
  return acc.self();
}

static void jfa_set_accVA2(bob::trainer::JFATrainer& trainer, 
  bob::python::const_ndarray acc)
{
  const blitz::Array<double,2> acc_ = acc.bz<double,2>();
  trainer.setAccVA2(acc_);
}

static object jfa_get_accDA1(const bob::trainer::JFATrainer& trainer)
{
  const blitz::Array<double,1>& acc_ref = trainer.getAccDA1();
  bob::python::ndarray acc(bob::core::array::t_float64, acc_ref.extent(0));
  blitz::Array<double,1> acc_ = acc.bz<double,1>();
  acc_ = acc_ref;
  return acc.self();
}

static void jfa_set_accDA1(bob::trainer::JFATrainer& trainer, 
  bob::python::const_ndarray acc)
{
  const blitz::Array<double,1> acc_ = acc.bz<double,1>();
  trainer.setAccDA1(acc_);
}

static object jfa_get_accDA2(const bob::trainer::JFATrainer& trainer)
{
  const blitz::Array<double,1>& acc_ref = trainer.getAccDA2();
  bob::python::ndarray acc(bob::core::array::t_float64, acc_ref.extent(0));
  blitz::Array<double,1> acc_ = acc.bz<double,1>();
  acc_ = acc_ref;
  return acc.self();
}

static void jfa_set_accDA2(bob::trainer::JFATrainer& trainer, 
  bob::python::const_ndarray acc)
{
  const blitz::Array<double,1> acc_ = acc.bz<double,1>();
  trainer.setAccDA2(acc_);
}



void bind_trainer_jfa() 
{
  class_<bob::trainer::ISVTrainer, boost::noncopyable >("ISVTrainer", "Create a trainer for the ISV.", init<optional<const size_t, const double> >((arg("max_iterations")=10, arg("relevance_factor")=4.),"Initializes a new ISVTrainer."))
    .def(init<const bob::trainer::ISVTrainer&>((arg("other")), "Copy constructs an ISVTrainer"))
    .add_property("max_iterations", &bob::trainer::ISVTrainer::getMaxIterations, &bob::trainer::ISVTrainer::setMaxIterations, "Max iterations")
    .add_property("rng", &bob::trainer::ISVTrainer::getRng, &bob::trainer::ISVTrainer::setRng, "The Mersenne Twister mt19937 random generator used for the initialization of subspaces/arrays before the EM loop.")
    .add_property("__X__", &isv_get_x, &isv_set_x)
    .add_property("__Z__", &isv_get_z, &isv_set_z)
    .def(self == self)
    .def(self != self)
    .def("is_similar_to", &bob::trainer::ISVTrainer::is_similar_to, (arg("self"), arg("other"), arg("r_epsilon")=1e-5, arg("a_epsilon")=1e-8), "Compares this ISVTrainer with the 'other' one to be approximately the same.")
    .def("train", &isv_train, (arg("self"), arg("isv_base"), arg("gmm_stats")), "Call the training procedure.")
    .def("initialize", &isv_initialize, (arg("self"), arg("isv_base"), arg("gmm_stats")), "Call the initialization procedure.")
    .def("e_step", &isv_estep, (arg("self"), arg("isv_base"), arg("gmm_stats")), "Call the e-step procedure.")
    .def("m_step", &isv_mstep, (arg("self"), arg("isv_base"), arg("gmm_stats")), "Call the m-step procedure.")
    .def("finalize", &isv_finalize, (arg("self"), arg("isv_base"), arg("gmm_stats")), "Call the finalization procedure.")
    .def("enrol", &isv_enrol, (arg("self"), arg("isv_machine"), arg("gmm_stats"), arg("n_iter")), "Call the enrolment procedure.")
    .add_property("acc_u_a1", &isv_get_accUA1, &isv_set_accUA1, "Accumulator updated during the E-step")
    .add_property("acc_u_a2", &isv_get_accUA2, &isv_set_accUA2, "Accumulator updated during the E-step")
  ;

  class_<bob::trainer::JFATrainer, boost::noncopyable >("JFATrainer", "Create a trainer for the ISV.", init<optional<const size_t> >((arg("max_iterations")=10),"Initializes a new JFATrainer."))
    .def(init<const bob::trainer::JFATrainer&>((arg("other")), "Copy constructs an JFATrainer"))
    .add_property("max_iterations", &bob::trainer::JFATrainer::getMaxIterations, &bob::trainer::JFATrainer::setMaxIterations, "Max iterations")
    .add_property("rng", &bob::trainer::JFATrainer::getRng, &bob::trainer::JFATrainer::setRng, "The Mersenne Twister mt19937 random generator used for the initialization of subspaces/arrays before the EM loop.")
    .add_property("__X__", &jfa_get_x, &jfa_set_x)
    .add_property("__Y__", &jfa_get_y, &jfa_set_y)
    .add_property("__Z__", &jfa_get_z, &jfa_set_z)
    .def(self == self)
    .def(self != self)
    .def("is_similar_to", &bob::trainer::JFATrainer::is_similar_to, (arg("self"), arg("other"), arg("r_epsilon")=1e-5, arg("a_epsilon")=1e-8), "Compares this JFATrainer with the 'other' one to be approximately the same.")
    .def("train", &jfa_train, (arg("self"), arg("jfa_base"), arg("gmm_stats")), "Call the training procedure.")
    .def("initialize", &jfa_initialize, (arg("self"), arg("jfa_base"), arg("gmm_stats")), "Call the initialization procedure.")
    .def("train_loop", &jfa_train_loop, (arg("self"), arg("jfa_base"), arg("gmm_stats")), "Call the training procedure (without the initialization). This will train the three subspaces U, V and d.")
    .def("e_step1", &jfa_estep1, (arg("self"), arg("jfa_base"), arg("gmm_stats")), "Call the 1st e-step procedure (for the V subspace).")
    .def("m_step1", &jfa_mstep1, (arg("self"), arg("jfa_base"), arg("gmm_stats")), "Call the 1st m-step procedure (for the V subspace).")
    .def("finalize1", &jfa_finalize1, (arg("self"), arg("jfa_base"), arg("gmm_stats")), "Call the 1st finalization procedure (for the V subspace).")
    .def("e_step2", &jfa_estep2, (arg("self"), arg("jfa_base"), arg("gmm_stats")), "Call the 2nd e-step procedure (for the U subspace).")
    .def("m_step2", &jfa_mstep2, (arg("self"), arg("jfa_base"), arg("gmm_stats")), "Call the 2nd m-step procedure (for the U subspace).")
    .def("finalize2", &jfa_finalize2, (arg("self"), arg("jfa_base"), arg("gmm_stats")), "Call the 2nd finalization procedure (for the U subspace).")
    .def("e_step3", &jfa_estep3, (arg("self"), arg("jfa_base"), arg("gmm_stats")), "Call the 3rd e-step procedure (for the d subspace).")
    .def("m_step3", &jfa_mstep3, (arg("self"), arg("jfa_base"), arg("gmm_stats")), "Call the 3rd m-step procedure (for the d subspace).")
    .def("finalize3", &jfa_finalize3, (arg("self"), arg("jfa_base"), arg("gmm_stats")), "Call the 3rd finalization procedure (for the d subspace).")
    .def("enrol", &jfa_enrol, (arg("self"), arg("jfa_machine"), arg("gmm_stats"), arg("n_iter")), "Call the enrolment procedure.")
    .add_property("acc_v_a1", &jfa_get_accVA1, &jfa_set_accVA1, "Accumulator updated during the E-step")
    .add_property("acc_v_a2", &jfa_get_accVA2, &jfa_set_accVA2, "Accumulator updated during the E-step")
    .add_property("acc_u_a1", &jfa_get_accUA1, &jfa_set_accUA1, "Accumulator updated during the E-step")
    .add_property("acc_u_a2", &jfa_get_accUA2, &jfa_set_accUA2, "Accumulator updated during the E-step")
    .add_property("acc_d_a1", &jfa_get_accDA1, &jfa_set_accDA1, "Accumulator updated during the E-step")
    .add_property("acc_d_a2", &jfa_get_accDA2, &jfa_set_accDA2, "Accumulator updated during the E-step")
  ;
}
