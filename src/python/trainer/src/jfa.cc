/**
 * @file python/trainer/src/jfa.cc
 * @date Tue Jul 19 12:16:17 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Python bindings to Joint Factor Analysis trainers
 *
 * Copyright (C) 2011-2012 Idiap Reasearch Institute, Martigny, Switzerland
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

#include "core/python/ndarray.h"
#include <boost/python/stl_iterator.hpp>
#include "trainer/JFATrainer.h"
#include "machine/JFAMachine.h"

using namespace boost::python;
namespace train = bob::trainer;
namespace mach = bob::machine;
namespace tp = bob::python;
namespace ca = bob::core::array;

static void jfa_train(train::JFABaseTrainer& t, list list_stats, const size_t n_iter)
{
  int n_ids = len(list_stats);
  std::vector<std::vector<const bob::machine::GMMStats*> > gmm_stats;

  // Extracts the vector of vector of pointers from the python list of lists
  for(int id=0; id<n_ids; ++id) {
    list list_stats_id = extract<list>(list_stats[id]);
    int n_samples = len(list_stats_id);
    std::vector<const bob::machine::GMMStats*> gmm_stats_id;
    for(int s=0; s<n_samples; ++s)
      gmm_stats_id.push_back(extract<const bob::machine::GMMStats*>(list_stats_id[s]));
    gmm_stats.push_back(gmm_stats_id);
  }

  // Calls the train function
  t.train(gmm_stats, n_iter);
}

static void jfa_train_vector(train::JFABaseTrainer& t, 
    object N, object F, size_t n_iter) {
  //N
  stl_input_iterator<tp::const_ndarray> it(N), end;
  std::vector<blitz::Array<double,2> > Nref;
  Nref.reserve(len(N));
  for (; it != end; ++it) Nref.push_back((*it).bz<double,2>());

  //F
  stl_input_iterator<tp::const_ndarray> it2(F);
  std::vector<blitz::Array<double,2> > Fref;
  Fref.reserve(len(F));
  for (; it2 != end; ++it2) Fref.push_back((*it2).bz<double,2>());

  t.train(Nref, Fref, n_iter);
}

static void jfa_train_noinit(train::JFABaseTrainer& t, list list_stats, const size_t n_iter)
{
  int n_ids = len(list_stats);
  std::vector<std::vector<const bob::machine::GMMStats*> > gmm_stats;

  // Extracts the vector of vector of pointers from the python list of lists
  for(int id=0; id<n_ids; ++id) {
    list list_stats_id = extract<list>(list_stats[id]);
    int n_samples = len(list_stats_id);
    std::vector<const bob::machine::GMMStats*> gmm_stats_id;
    for(int s=0; s<n_samples; ++s)
      gmm_stats_id.push_back(extract<const bob::machine::GMMStats*>(list_stats_id[s]));
    gmm_stats.push_back(gmm_stats_id);
  }

  // Calls the train function
  t.trainNoInit(gmm_stats, n_iter);
}


static void jfa_train_ISV(train::JFABaseTrainer& t, list list_stats, 
  const size_t n_iter, const double relevance_factor)
{
  int n_ids = len(list_stats);
  std::vector<std::vector<const bob::machine::GMMStats*> > gmm_stats;

  // Extracts the vector of vector of pointers from the python list of lists
  for(int id=0; id<n_ids; ++id) {
    list list_stats_id = extract<list>(list_stats[id]);
    int n_samples = len(list_stats_id);
    std::vector<const bob::machine::GMMStats*> gmm_stats_id;
    for(int s=0; s<n_samples; ++s)
      gmm_stats_id.push_back(extract<const bob::machine::GMMStats*>(list_stats_id[s]));
    gmm_stats.push_back(gmm_stats_id);
  }

  // Calls the train function
  t.trainISV(gmm_stats, n_iter, relevance_factor);
}

static void jfa_train_ISV_noinit(train::JFABaseTrainer& t, list list_stats, 
  const size_t n_iter, const double relevance_factor)
{
  int n_ids = len(list_stats);
  std::vector<std::vector<const bob::machine::GMMStats*> > gmm_stats;

  // Extracts the vector of vector of pointers from the python list of lists
  for(int id=0; id<n_ids; ++id) {
    list list_stats_id = extract<list>(list_stats[id]);
    int n_samples = len(list_stats_id);
    std::vector<const bob::machine::GMMStats*> gmm_stats_id;
    for(int s=0; s<n_samples; ++s)
      gmm_stats_id.push_back(extract<const bob::machine::GMMStats*>(list_stats_id[s]));
    gmm_stats.push_back(gmm_stats_id);
  }

  // Calls the train function
  t.trainISVNoInit(gmm_stats, n_iter, relevance_factor);
}

static void jfa_enrol(train::JFATrainer& t, list stats, const size_t n_iter)
{
  int n_samples = len(stats);
  std::vector<const bob::machine::GMMStats*> gmm_stats;
  for(int s=0; s<n_samples; ++s)
    gmm_stats.push_back(extract<const bob::machine::GMMStats*>(stats[s]));

  // Calls the enrol function
  t.enrol(gmm_stats, n_iter);
}

static void update_eigen(tp::const_ndarray A, tp::const_ndarray C, 
    tp::ndarray uv) {
  blitz::Array<double,2> uv_ = uv.bz<double,2>();
  train::jfa::updateEigen(A.bz<double,3>(), C.bz<double,2>(), uv_);
}

static void estimate_xandu(tp::const_ndarray F, tp::const_ndarray N,
    tp::const_ndarray m, tp::const_ndarray E,
    tp::const_ndarray d, tp::const_ndarray v,
    tp::const_ndarray u, tp::const_ndarray z,
    tp::const_ndarray y, tp::ndarray x,
    tp::const_ndarray spk_ids) {
  blitz::Array<double,2> x_ = x.bz<double,2>();
  train::jfa::estimateXandU(F.bz<double,2>(), N.bz<double,2>(),
      m.bz<double,1>(), E.bz<double,1>(), d.bz<double,1>(), v.bz<double,2>(),
      u.bz<double,2>(), z.bz<double,2>(), y.bz<double,2>(), x_,
      spk_ids.bz<uint32_t,1>());
}

static void estimate_yandv(tp::const_ndarray F, tp::const_ndarray N,
  tp::const_ndarray m, tp::const_ndarray E, 
  tp::const_ndarray d, tp::const_ndarray v, 
  tp::const_ndarray u, tp::const_ndarray z, 
  tp::ndarray y, tp::const_ndarray x, tp::const_ndarray spk_ids) {
  blitz::Array<double,2> y_ = y.bz<double,2>();
  train::jfa::estimateYandV(F.bz<double,2>(), N.bz<double,2>(),
      m.bz<double,1>(), E.bz<double,1>(), d.bz<double,1>(), v.bz<double,2>(),
      u.bz<double,2>(), z.bz<double,2>(), y_, x.bz<double,2>(), 
      spk_ids.bz<uint32_t,1>());
}

static void estimate_zandd(tp::const_ndarray F, tp::const_ndarray N,
  tp::const_ndarray m, tp::const_ndarray E,
  tp::const_ndarray d, tp::const_ndarray v,
  tp::const_ndarray u, tp::ndarray z,
  tp::const_ndarray y, tp::const_ndarray x,
  tp::const_ndarray spk_ids) {
  blitz::Array<double,2> z_ = z.bz<double,2>();
  train::jfa::estimateZandD(F.bz<double,2>(), N.bz<double,2>(),
      m.bz<double,1>(), E.bz<double,1>(), d.bz<double,1>(), v.bz<double,2>(),
      u.bz<double,2>(), z_, y.bz<double,2>(), x.bz<double,2>(),
      spk_ids.bz<uint32_t,1>());
}

template <typename T, int N>
tuple as_tuple (const std::vector<blitz::Array<T,N> >& obj) {
  list retval;
  for (size_t k=0; k<obj.size(); ++k) retval.append(obj[k]); //copy
  return tuple(retval);
}

static tuple get_n (const train::JFABaseTrainer& obj) {
  return as_tuple(obj.getN()); 
}

static tuple get_f (const train::JFABaseTrainer& obj) {
  return as_tuple(obj.getF());
}

static tuple get_x (const train::JFABaseTrainer& obj) {
  return as_tuple(obj.getX());
}

static tuple get_y (const train::JFABaseTrainer& obj) {
  return as_tuple(obj.getY());
}

static tuple get_z (const train::JFABaseTrainer& obj) {
  return as_tuple(obj.getZ());
}

static void jfa_set_stats(train::JFABaseTrainer& t, object N,
    object F) {
  //N
  stl_input_iterator<tp::const_ndarray> it(N), end;
  std::vector<blitz::Array<double,2> > Nref;
  Nref.reserve(len(N));
  for (; it != end; ++it) Nref.push_back((*it).bz<double,2>());

  //F
  stl_input_iterator<tp::const_ndarray> it2(F);
  std::vector<blitz::Array<double,2> > Fref;
  Fref.reserve(len(F));
  for (; it2 != end; ++it2) Fref.push_back((*it2).bz<double,2>());

  t.setStatistics(Nref, Fref);
}

static void jfa_set_speaker_factors(train::JFABaseTrainer& t, 
    object x, object y, object z) {
  //x
  stl_input_iterator<tp::const_ndarray> it(x), end;
  std::vector<blitz::Array<double,2> > xref;
  xref.reserve(len(x));
  for (; it != end; ++it) xref.push_back((*it).bz<double,2>());

  //y
  stl_input_iterator<tp::const_ndarray> it2(y);
  std::vector<blitz::Array<double,1> > yref;
  yref.reserve(len(y));
  for (; it2 != end; ++it2) yref.push_back((*it2).bz<double,1>());

  //z
  stl_input_iterator<tp::const_ndarray> it3(z);
  std::vector<blitz::Array<double,1> > zref;
  zref.reserve(len(z));
  for (; it3 != end; ++it3) zref.push_back((*it3).bz<double,1>());

  t.setSpeakerFactors(xref, yref, zref);
}

void bind_trainer_jfa() {
  def("jfa_updateEigen", &update_eigen, (arg("A"), arg("C"), arg("uv")), "Updates eigenchannels (or eigenvoices) from accumulators A and C.");
  def("jfa_estimateXandU", &estimate_xandu, (arg("F"), arg("N"), arg("m"), arg("E"), arg("d"), arg("v"), arg("u"), arg("z"), arg("y"), arg("x"), arg("spk_ids")), "Estimates the channel factors.");
  def("jfa_estimateYandV", &estimate_yandv, (arg("F"), arg("N"), arg("m"), arg("E"), arg("d"), arg("v"), arg("u"), arg("z"), arg("y"), arg("x"), arg("spk_ids")), "Estimates the speaker factors y.");
  def("jfa_estimateZandD", &estimate_zandd, (arg("F"), arg("N"), arg("m"), arg("E"), arg("d"), arg("v"), arg("u"), arg("z"), arg("y"), arg("x"), arg("spk_ids")), "Estimates the speaker factors z.");

  class_<train::JFABaseTrainer, boost::noncopyable>("JFABaseTrainer", "Create a trainer for the JFA.", init<mach::JFABaseMachine&>((arg("jfa_base")),"Initializes a new JFABaseTrainer."))

    .add_property("N", &get_n, &train::JFABaseTrainer::setN)
    .add_property("F", &get_f, &train::JFABaseTrainer::setF)
    .add_property("X", &get_x, &train::JFABaseTrainer::setX)
    .add_property("Y", &get_y, &train::JFABaseTrainer::setY)
    .add_property("Z", &get_z, &train::JFABaseTrainer::setZ)

    .add_property("VtSigmaInv", make_function(&train::JFABaseTrainer::getVtSigmaInv, return_value_policy<copy_const_reference>()), &train::JFABaseTrainer::setVtSigmaInv)
    .add_property("IdPlusVProd_i", make_function(&train::JFABaseTrainer::getIdPlusVProd_i, return_value_policy<copy_const_reference>()), &train::JFABaseTrainer::setIdPlusVProd_i)
    .add_property("Fn_y_i", make_function(&train::JFABaseTrainer::getFn_y_i, return_value_policy<copy_const_reference>()), &train::JFABaseTrainer::setFn_y_i)
    .add_property("A1_y", make_function(&train::JFABaseTrainer::getA1_y, return_value_policy<copy_const_reference>()), &train::JFABaseTrainer::setA1_y)
    .add_property("A2_y", make_function(&train::JFABaseTrainer::getA2_y, return_value_policy<copy_const_reference>()), &train::JFABaseTrainer::setA2_y)
    .def("setStatistics", &jfa_set_stats, (arg("self"), arg("N"), arg("F")), "Set the zeroth and first order statistics.")
    .def("setSpeakerFactors", &jfa_set_speaker_factors, (arg("self"), arg("x"), arg("y"), arg("z")), "Set the speaker factors.")
    .def("train", &jfa_train_vector, (arg("self"), arg("N"), arg("F"), arg("n_iter")), "Call the training procedure.")
    .def("train", &jfa_train, (arg("self"), arg("gmm_stats"), arg("n_iter")), "Call the training procedure.")
    .def("trainNoInit", &jfa_train_noinit, (arg("self"), arg("gmm_stats"), arg("n_iter")), "Call the training procedure.")
    .def("trainISV", &jfa_train_ISV, (arg("self"), arg("gmm_stats"), arg("n_iter"), arg("relevance")), "Call the ISV training procedure.")
    .def("trainISVNoInit", &jfa_train_ISV_noinit, (arg("self"), arg("gmm_stats"), arg("n_iter"), arg("relevance")), "Call the ISV training procedure.")
    .def("initializeUVD", &train::JFABaseTrainer::initializeUVD, (arg("self")), "Initializes randomly U, V and D.")
    .def("initializeVD_ISV", &train::JFABaseTrainer::initializeVD_SIV, (arg("self"), arg("relevance factor")), "Initializes V=0 and D=sqrt(var(UBM)/r) (for ISV).")
    .def("initializeRandomU", &train::JFABaseTrainer::initializeRandomU, (arg("self")), "Initializes randomly U.")
    .def("initializeRandomV", &train::JFABaseTrainer::initializeRandomV, (arg("self")), "Initializes randomly V.")
    .def("initializeRandomD", &train::JFABaseTrainer::initializeRandomD, (arg("self")), "Initializes randomly D.")
    .def("computeVtSigmaInv", &train::JFABaseTrainer::computeVtSigmaInv, (arg("self")), "Computes Vt*SigmaInv.")
    .def("computeVProd", &train::JFABaseTrainer::computeVProd, (arg("self")), "Computes VProd.")
    .def("computeIdPlusVProd_i", &train::JFABaseTrainer::computeIdPlusVProd_i, (arg("self"), arg("id")), "Computes IdPlusVProd_i.")
    .def("computeFn_y_i", &train::JFABaseTrainer::computeFn_y_i, (arg("self"), arg("id")), "Computes Fn_y_i.")
    .def("updateY_i", &train::JFABaseTrainer::updateY_i, (arg("self"), arg("id")), "Updates Y_i.")
    .def("updateY", &train::JFABaseTrainer::updateY, (arg("self")), "Updates Y.")
    .def("updateV", &train::JFABaseTrainer::updateV, (arg("self")), "Updates V.")
    .def("computeUtSigmaInv", &train::JFABaseTrainer::computeUtSigmaInv, (arg("self")), "Computes Ut*SigmaInv.")
    .def("computeIdPlusUProd_ih", &train::JFABaseTrainer::computeIdPlusUProd_ih, (arg("self"), arg("id"), arg("h")), "Computes IdPlusUProd_ih.")
    .def("computeFn_x_ih", &train::JFABaseTrainer::computeFn_x_ih, (arg("self"), arg("id"), arg("h")), "Computes Fn_x_ih.")
    .def("updateX_ih", &train::JFABaseTrainer::updateX_ih, (arg("self"), arg("id"), arg("h")), "Updates X_ih.")
    .def("updateX", &train::JFABaseTrainer::updateX, (arg("self")), "Updates X.")
    .def("updateU", &train::JFABaseTrainer::updateU, (arg("self")), "Updates U.")
    .def("computeDtSigmaInv", &train::JFABaseTrainer::computeDtSigmaInv, (arg("self")), "Computes Dt*SigmaInv.")
    .def("computeIdPlusDProd_i", &train::JFABaseTrainer::computeIdPlusDProd_i, (arg("self"), arg("id")), "Computes IdPlusDProd_i.")
    .def("computeFn_z_i", &train::JFABaseTrainer::computeFn_z_i, (arg("self"), arg("id")), "Computes Fn_z_i.")
    .def("updateZ_i", &train::JFABaseTrainer::updateZ_i, (arg("self"), arg("id")), "Updates Z_i.")
    .def("updateZ", &train::JFABaseTrainer::updateZ, (arg("self")), "Updates Z.")
    .def("updateD", &train::JFABaseTrainer::updateD, (arg("self")), "Updates D.")
    .def("precomputeSumStatisticsN", &train::JFABaseTrainer::precomputeSumStatisticsN, (arg("self")), "Precomputes zeroth order statistics over sessions.")
    .def("precomputeSumStatisticsF", &train::JFABaseTrainer::precomputeSumStatisticsF, (arg("self")), "Precomputes first order statistics over sessions.")
    ;

  class_<train::JFATrainer, boost::noncopyable>("JFATrainer", "Create a trainer for the JFA.", init<mach::JFAMachine&, train::JFABaseTrainer&>((arg("jfa"), arg("base_trainer")),"Initializes a new JFATrainer."))
    .def("enrol", (void (train::JFATrainer::*)(const blitz::Array<double,2>&, const blitz::Array<double,2>&, const size_t))&train::JFATrainer::enrol, (arg("self"), arg("N"), arg("F"), arg("n_iter")), "Call the training procedure.")
    .def("enrol", &jfa_enrol, (arg("self"), arg("gmm_stats"), arg("n_iter")), "Call the training procedure.")
    ;


}
