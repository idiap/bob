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

#include <bob/core/python/ndarray.h>
#include <boost/python/stl_iterator.hpp>
#include <bob/trainer/JFATrainer.h>
#include <boost/shared_ptr.hpp>

using namespace boost::python;

static void update_eigen(bob::python::const_ndarray A, bob::python::const_ndarray C, 
    bob::python::ndarray uv) {
  blitz::Array<double,2> uv_ = uv.bz<double,2>();
  bob::trainer::jfa::updateEigen(A.bz<double,3>(), C.bz<double,2>(), uv_);
}

static void estimate_xandu(bob::python::const_ndarray F, bob::python::const_ndarray N,
    bob::python::const_ndarray m, bob::python::const_ndarray E,
    bob::python::const_ndarray d, bob::python::const_ndarray v,
    bob::python::const_ndarray u, bob::python::const_ndarray z,
    bob::python::const_ndarray y, bob::python::ndarray x,
    bob::python::const_ndarray spk_ids) {
  blitz::Array<double,2> x_ = x.bz<double,2>();
  bob::trainer::jfa::estimateXandU(F.bz<double,2>(), N.bz<double,2>(),
      m.bz<double,1>(), E.bz<double,1>(), d.bz<double,1>(), v.bz<double,2>(),
      u.bz<double,2>(), z.bz<double,2>(), y.bz<double,2>(), x_,
      spk_ids.bz<uint32_t,1>());
}

static void estimate_yandv(bob::python::const_ndarray F, bob::python::const_ndarray N,
  bob::python::const_ndarray m, bob::python::const_ndarray E, 
  bob::python::const_ndarray d, bob::python::const_ndarray v, 
  bob::python::const_ndarray u, bob::python::const_ndarray z, 
  bob::python::ndarray y, bob::python::const_ndarray x, bob::python::const_ndarray spk_ids) {
  blitz::Array<double,2> y_ = y.bz<double,2>();
  bob::trainer::jfa::estimateYandV(F.bz<double,2>(), N.bz<double,2>(),
      m.bz<double,1>(), E.bz<double,1>(), d.bz<double,1>(), v.bz<double,2>(),
      u.bz<double,2>(), z.bz<double,2>(), y_, x.bz<double,2>(), 
      spk_ids.bz<uint32_t,1>());
}

static void estimate_zandd(bob::python::const_ndarray F, bob::python::const_ndarray N,
  bob::python::const_ndarray m, bob::python::const_ndarray E,
  bob::python::const_ndarray d, bob::python::const_ndarray v,
  bob::python::const_ndarray u, bob::python::ndarray z,
  bob::python::const_ndarray y, bob::python::const_ndarray x,
  bob::python::const_ndarray spk_ids) {
  blitz::Array<double,2> z_ = z.bz<double,2>();
  bob::trainer::jfa::estimateZandD(F.bz<double,2>(), N.bz<double,2>(),
      m.bz<double,1>(), E.bz<double,1>(), d.bz<double,1>(), v.bz<double,2>(),
      u.bz<double,2>(), z_, y.bz<double,2>(), x.bz<double,2>(),
      spk_ids.bz<uint32_t,1>());
}

static void extractGMMStatsVectors(list list_stats, 
  std::vector<std::vector<boost::shared_ptr<const bob::machine::GMMStats> > >& gmm_stats)
{
  const size_t n_ids = len(list_stats);
  // Extracts the vector of vector of pointers from the python list of lists
  for(size_t id=0; id<n_ids; ++id) {
    list list_stats_id = extract<list>(list_stats[id]);
    size_t n_samples = len(list_stats_id);
    std::vector<boost::shared_ptr<const bob::machine::GMMStats> > gmm_stats_id;
    for(size_t s=0; s<n_samples; ++s) {
      boost::shared_ptr<bob::machine::GMMStats> gs = extract<boost::shared_ptr<bob::machine::GMMStats> >(list_stats_id[s]);
      gmm_stats_id.push_back(gs);
    }
    gmm_stats.push_back(gmm_stats_id);
  }
}

static void jfa_train(bob::trainer::JFABaseTrainer& t, list list_stats, const size_t n_iter)
{
  std::vector<std::vector<boost::shared_ptr<const bob::machine::GMMStats> > > gmm_stats;
  extractGMMStatsVectors(list_stats, gmm_stats);
  // Calls the train function
  t.train(gmm_stats, n_iter);
}

static void jfa_train_noinit(bob::trainer::JFABaseTrainer& t, list list_stats, const size_t n_iter)
{
  std::vector<std::vector<boost::shared_ptr<const bob::machine::GMMStats> > > gmm_stats;
  extractGMMStatsVectors(list_stats, gmm_stats);
  // Calls the train function
  t.trainNoInit(gmm_stats, n_iter);
}


static void jfa_train_ISV(bob::trainer::JFABaseTrainer& t, list list_stats, 
  const size_t n_iter, const double relevance_factor)
{
  std::vector<std::vector<boost::shared_ptr<const bob::machine::GMMStats> > > gmm_stats;
  extractGMMStatsVectors(list_stats, gmm_stats);
  // Calls the train function
  t.trainISV(gmm_stats, n_iter, relevance_factor);
}

static void jfa_train_ISV_noinit(bob::trainer::JFABaseTrainer& t, list list_stats, 
  const size_t n_iter, const double relevance_factor)
{
  std::vector<std::vector<boost::shared_ptr<const bob::machine::GMMStats> > > gmm_stats;
  extractGMMStatsVectors(list_stats, gmm_stats);
  // Calls the train function
  t.trainISVNoInit(gmm_stats, n_iter, relevance_factor);
}

static void jfa_enrol(bob::trainer::JFATrainer& t, list stats, const size_t n_iter)
{
  int n_samples = len(stats);
  std::vector<boost::shared_ptr<const bob::machine::GMMStats> > gmm_stats;
  for(int s=0; s<n_samples; ++s)
  {
    boost::shared_ptr<bob::machine::GMMStats> gs = extract<boost::shared_ptr<bob::machine::GMMStats> >(stats[s]);  
    gmm_stats.push_back(gs);
  }

  // Calls the enrol function
  t.enrol(gmm_stats, n_iter);
}


template <typename T, int N>
tuple as_tuple (const std::vector<blitz::Array<T,N> >& obj) {
  list retval;
  for (size_t k=0; k<obj.size(); ++k) retval.append(obj[k]); //copy
  return tuple(retval);
}

static tuple get_x (const bob::trainer::JFABaseTrainerBase& obj) {
  return as_tuple(obj.getX());
}

static tuple get_y (const bob::trainer::JFABaseTrainerBase& obj) {
  return as_tuple(obj.getY());
}

static tuple get_z (const bob::trainer::JFABaseTrainerBase& obj) {
  return as_tuple(obj.getZ());
}

static void jfa_set_speaker_factors(bob::trainer::JFABaseTrainerBase& t, 
    object x, object y, object z) {
  //x
  stl_input_iterator<bob::python::const_ndarray> it(x), end;
  std::vector<blitz::Array<double,2> > xref;
  xref.reserve(len(x));
  for (; it != end; ++it) xref.push_back((*it).bz<double,2>());

  //y
  stl_input_iterator<bob::python::const_ndarray> it2(y);
  std::vector<blitz::Array<double,1> > yref;
  yref.reserve(len(y));
  for (; it2 != end; ++it2) yref.push_back((*it2).bz<double,1>());

  //z
  stl_input_iterator<bob::python::const_ndarray> it3(z);
  std::vector<blitz::Array<double,1> > zref;
  zref.reserve(len(z));
  for (; it3 != end; ++it3) zref.push_back((*it3).bz<double,1>());

  t.setSpeakerFactors(xref, yref, zref);
}

static void jfa_initNid(bob::trainer::JFABaseTrainerBase& t, object o)
{
  size_t Nid;
  extract<int> int_check(o);
  extract<double> float_check(o);
  if(int_check.check()) { //is int
    Nid = int_check();
  }
  else if(float_check.check()) { //is float
    Nid = static_cast<size_t>(float_check());
  }
  else {
    Nid = len(o);
  }
  t.initNid(Nid);
}


static void jfa_precomputeN(bob::trainer::JFABaseTrainerBase& t, list list_stats)
{
  std::vector<std::vector<boost::shared_ptr<const bob::machine::GMMStats> > > gmm_stats;
  extractGMMStatsVectors(list_stats, gmm_stats);
  // Calls the train function
  t.precomputeSumStatisticsN(gmm_stats);
}

static void jfa_precomputeF(bob::trainer::JFABaseTrainerBase& t, list list_stats)
{
  std::vector<std::vector<boost::shared_ptr<const bob::machine::GMMStats> > > gmm_stats;
  extractGMMStatsVectors(list_stats, gmm_stats);
  // Calls the train function
  t.precomputeSumStatisticsF(gmm_stats);
}

static void jfa_updateX(bob::trainer::JFABaseTrainer& t, list list_stats)
{
  std::vector<std::vector<boost::shared_ptr<const bob::machine::GMMStats> > > gmm_stats;
  extractGMMStatsVectors(list_stats, gmm_stats);
  // Calls the train function
  t.updateX(gmm_stats);
}

static void jfa_updateY(bob::trainer::JFABaseTrainer& t, list list_stats)
{
  std::vector<std::vector<boost::shared_ptr<const bob::machine::GMMStats> > > gmm_stats;
  extractGMMStatsVectors(list_stats, gmm_stats);
  // Calls the train function
  t.updateY(gmm_stats);
}

static void jfa_updateZ(bob::trainer::JFABaseTrainer& t, list list_stats)
{
  std::vector<std::vector<boost::shared_ptr<const bob::machine::GMMStats> > > gmm_stats;
  extractGMMStatsVectors(list_stats, gmm_stats);
  // Calls the train function
  t.updateZ(gmm_stats);
}

static void jfa_updateU(bob::trainer::JFABaseTrainer& t, list list_stats)
{
  std::vector<std::vector<boost::shared_ptr<const bob::machine::GMMStats> > > gmm_stats;
  extractGMMStatsVectors(list_stats, gmm_stats);
  // Calls the train function
  t.updateU(gmm_stats);
}

static void jfa_updateV(bob::trainer::JFABaseTrainer& t, list list_stats)
{
  std::vector<std::vector<boost::shared_ptr<const bob::machine::GMMStats> > > gmm_stats;
  extractGMMStatsVectors(list_stats, gmm_stats);
  // Calls the train function
  t.updateV(gmm_stats);
}

static void jfa_updateD(bob::trainer::JFABaseTrainer& t, list list_stats)
{
  std::vector<std::vector<boost::shared_ptr<const bob::machine::GMMStats> > > gmm_stats;
  extractGMMStatsVectors(list_stats, gmm_stats);
  // Calls the train function
  t.updateD(gmm_stats);
}

void bind_trainer_jfa() {
  def("jfa_update_eigen", &update_eigen, (arg("a"), arg("c"), arg("uv")), "Updates eigenchannels (or eigenvoices) from accumulators a and c.");
  def("jfa_estimate_x_and_u", &estimate_xandu, (arg("f"), arg("n"), arg("m"), arg("e"), arg("d"), arg("v"), arg("u"), arg("z"), arg("y"), arg("x"), arg("spk_ids")), "Estimates the channel factors.");
  def("jfa_estimate_y_and_v", &estimate_yandv, (arg("f"), arg("n"), arg("m"), arg("e"), arg("d"), arg("v"), arg("u"), arg("z"), arg("y"), arg("x"), arg("spk_ids")), "Estimates the speaker factors y.");
  def("jfa_estimate_z_and_d", &estimate_zandd, (arg("f"), arg("n"), arg("m"), arg("e"), arg("d"), arg("v"), arg("u"), arg("z"), arg("y"), arg("x"), arg("spk_ids")), "Estimates the speaker factors z.");

  class_<bob::trainer::JFABaseTrainerBase, boost::noncopyable>("JFABaseTrainerBase", "Create a trainer for the JFA.", init<bob::machine::JFABaseMachine&>((arg("jfa_base")),"Initializes a new JFABaseTrainerBase."))
    .add_property("__X__", &get_x, &bob::trainer::JFABaseTrainerBase::setX)
    .add_property("__Y__", &get_y, &bob::trainer::JFABaseTrainerBase::setY)
    .add_property("__Z__", &get_z, &bob::trainer::JFABaseTrainerBase::setZ)
    .def("__setSpeakerFactors__", &jfa_set_speaker_factors, (arg("self"), arg("x"), arg("y"), arg("z")), "Set the speaker factors.")
    .def("__initializeRandomU__", &bob::trainer::JFABaseTrainerBase::initializeRandomU, (arg("self")), "Initializes randomly U.")
    .def("__initializeRandomV__", &bob::trainer::JFABaseTrainerBase::initializeRandomV, (arg("self")), "Initializes randomly V.")
    .def("__initializeRandomD__", &bob::trainer::JFABaseTrainerBase::initializeRandomD, (arg("self")), "Initializes randomly D.")
    .def("__initializeUVD__", &bob::trainer::JFABaseTrainerBase::initializeUVD, (arg("self")), "Initializes randomly U, V and D.")
    .def("__initNid__", &jfa_initNid, (arg("self"), arg("stats")), "Initializes the number of identities.")
    .def("__precomputeSumStatisticsN__", &jfa_precomputeN, (arg("self"), arg("stats")), "Precomputes zeroth order statistics over sessions.")
    .def("__precomputeSumStatisticsF__", &jfa_precomputeF, (arg("self"), arg("stats")), "Precomputes first order statistics over sessions.")
  ;


  class_<bob::trainer::JFABaseTrainer, boost::noncopyable, bases<bob::trainer::JFABaseTrainerBase> >("JFABaseTrainer", "Create a trainer for the JFA.", init<bob::machine::JFABaseMachine&>((arg("jfa_base")),"Initializes a new JFABaseTrainer."))
    .def("train", &jfa_train, (arg("self"), arg("gmm_stats"), arg("n_iter")), "Call the training procedure.")
    .def("train_no_init", &jfa_train_noinit, (arg("self"), arg("gmm_stats"), arg("n_iter")), "Call the training procedure.")
    .def("train_isv", &jfa_train_ISV, (arg("self"), arg("gmm_stats"), arg("n_iter"), arg("relevance")), "Call the ISV training procedure.")
    .def("train_isv_no_init", &jfa_train_ISV_noinit, (arg("self"), arg("gmm_stats"), arg("n_iter"), arg("relevance")), "Call the ISV training procedure.")
    .def("__initializeVD_ISV__", &bob::trainer::JFABaseTrainer::initializeVD_ISV, (arg("self"), arg("relevance factor")), "Initializes V=0 and D=sqrt(var(UBM)/r) (for ISV).")
    .def("__updateY__", &jfa_updateY, (arg("self"), arg("stats")), "Updates Y.")
    .def("__updateV__", &jfa_updateV, (arg("self"), arg("stats")), "Updates V.")
    .def("__updateX__", &jfa_updateX, (arg("self"), arg("stats")), "Updates X.")
    .def("__updateU__", &jfa_updateU, (arg("self"), arg("stats")), "Updates U.")
    .def("__updateZ__", &jfa_updateZ, (arg("self"), arg("stats")), "Updates Z.")
    .def("__updateD__", &jfa_updateD, (arg("self"), arg("stats")), "Updates D.")
    ;

  class_<bob::trainer::JFATrainer, boost::noncopyable>("JFATrainer", "Create a trainer for the JFA.", init<bob::machine::JFAMachine&, bob::trainer::JFABaseTrainer&>((arg("jfa"), arg("base_trainer")),"Initializes a new JFATrainer."))
    .def("enrol", &jfa_enrol, (arg("self"), arg("gmm_stats"), arg("n_iter")), "Call the training procedure.")
    ;


}
