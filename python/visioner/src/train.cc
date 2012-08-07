/**
 * @file python/visioner/src/train.cc
 * @date Tue 31 Jul 2012 15:32:18 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Model training bridge for Visioner
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

#include <fstream>

#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>

#include "core/python/ndarray.h"

#include "visioner/model/mdecoder.h"
#include "visioner/model/sampler.h"

namespace bp = boost::python;
namespace tp = bob::python;

/**
 * Determines if the input filename ends in ".gz"
 *
 * @param filename The name of the file to be analyzed.
 */
inline static bool is_dot_gz(const std::string& filename) {
   return boost::filesystem::extension(filename) == ".gz" ||
     boost::filesystem::extension(filename) == ".vbgz";
}

inline static bool is_dot_vbin(const std::string& filename) {
  return boost::filesystem::extension(filename) == ".vbin" ||
    boost::filesystem::extension(filename) == ".vbgz";
}

static bool train_model(bob::visioner::Model& model, 
    const bob::visioner::Sampler& training, 
    const bob::visioner::Sampler& validation) {

  const bob::visioner::param_t param = model.param();

  // Train the model using coarse-to-fine feature projection
  for (uint64_t p = 0; p <= param.m_projections;
      ++p, model.project()) {
    if (bob::visioner::make_trainer(param)->train(training, validation, model) == false) return false;
  }

  // OK
  return true;
}

static boost::shared_ptr<bob::visioner::Model> model_from_path(const std::string& path) {

  std::ios_base::openmode mode = std::ios_base::in;
  if (is_dot_gz(path) || is_dot_vbin(path)) mode |= std::ios_base::binary;
  std::ifstream file(path.c_str(), mode);
  boost::iostreams::filtering_istream ifs; ///< the input stream
  if (is_dot_gz(path)) 
    ifs.push(boost::iostreams::basic_gzip_decompressor<>());
  ifs.push(file);

  if (ifs.good() == false) {
    PYTHON_ERROR(IOError, "failed to load model parameters from file '%s'", path.c_str());
  }

  bob::visioner::param_t param;
  if (is_dot_vbin(path)) { //a binary file from visioner
    boost::archive::binary_iarchive ia(ifs);
    ia >> param;
  }
  else { //the default
    boost::archive::text_iarchive ia(ifs);
    ia >> param;
  }

  if (!ifs.good()) {
    PYTHON_ERROR(IOError, "file '%s' finished before I could load the model - it seems only to contain parameters", path.c_str());
  }

  return bob::visioner::make_model(param);
}

/**
 * Listings of implemented stuff
 */
static bp::tuple available_losses() {
  bp::list retval;
  BOOST_FOREACH(const std::string& s, bob::visioner::available_losses_list()) retval.append(s);
  return bp::tuple(retval);
}

static bp::tuple available_taggers() {
  bp::list retval;
  BOOST_FOREACH(const std::string& s, bob::visioner::available_taggers_list()) retval.append(s);
  return bp::tuple(retval);
}

static bp::tuple available_models() {
  bp::list retval;
  BOOST_FOREACH(const std::string& s, bob::visioner::available_models_list()) retval.append(s);
  return bp::tuple(retval);
}

static bp::tuple available_trainers() {
  bp::list retval;
  BOOST_FOREACH(const std::string& s, bob::visioner::available_trainers_list()) retval.append(s);
  return bp::tuple(retval);
}

static bp::tuple available_optimizations() {
  bp::list retval;
  BOOST_FOREACH(const std::string& s, bob::visioner::available_optimizations_list()) retval.append(s);
  return bp::tuple(retval);
}

static bp::tuple available_sharings() {
  bp::list retval;
  BOOST_FOREACH(const std::string& s, bob::visioner::available_sharings_list()) retval.append(s);
  return bp::tuple(retval);
}

static void sampler_load(bob::visioner::Sampler& s, bp::object images,
    bp::object gts) {
  bp::stl_input_iterator<const char*> ibegin(images), iend;
  std::vector<std::string> i(ibegin, iend);
    bp::stl_input_iterator<const char*> gbegin(gts), gend;
  std::vector<std::string> g(gbegin, gend);
  s.load(i, g);
}

static boost::shared_ptr<bob::visioner::Sampler> 
sampler_from_files(const bob::visioner::param_t& param,
    bob::visioner::Sampler::SamplerType type, bp::object images,
    bp::object gts) {
  boost::shared_ptr<bob::visioner::Sampler> retval(new bob::visioner::Sampler(param, type));
  sampler_load(*retval, images, gts);
  return retval;
}

void bind_visioner_train() {
  bp::class_<bob::visioner::param_t>("param", "Various parameters useful for training boosted classifiers in the context of the Visioner", bp::init<bp::optional<uint64_t, uint64_t, const std::string&, double, const std::string&, const std::string&, uint64_t, const std::string&, const std::string&, uint64_t, double, uint64_t, const std::string&> >((bp::arg("rows")=24, bp::arg("cols")=20, bp::arg("loss")="diag_log", bp::arg("loss_parameter")=0.0, bp::arg("optimization_type")="ept", bp::arg("training_model")="gboost", bp::arg("num_of_bootstraps")=3, bp::arg("feature_type")="elbp", bp::arg("feature_sharing")="shared", bp::arg("feature_projections")=0, bp::arg("min_gt_overlap")=0.8, bp::arg("sliding_windows")=2, bp::arg("subwindow_labelling")="object_type"), "Default constructor. Note: The seed, number of training and validation samples, as well as the maximum number of boosting rounds is hard-coded."))
    .def_readwrite("rows", &bob::visioner::param_t::m_rows, "Number of rows in pixels")
    .def_readwrite("cols", &bob::visioner::param_t::m_cols, "Number of columns in pixels")
    .def_readwrite("seed", &bob::visioner::param_t::m_seed, "Random seed used for sampling")

    .def_readwrite("loss", &bob::visioner::param_t::m_loss, "Loss type")
    .def_readwrite("loss_parameter", &bob::visioner::param_t::m_loss_param, "Loss parameter")
    .def_readwrite("optimization_type", &bob::visioner::param_t::m_optimization, "Optimization type")
    .def_readwrite("training_model", &bob::visioner::param_t::m_trainer, "Training model")

    .def_readwrite("max_rounds", &bob::visioner::param_t::m_rounds, "Maximum number of boosting rounds")
    .def_readwrite("num_of_bootstraps", &bob::visioner::param_t::m_bootstraps, "Number of bootstrapping steps")

    .def_readwrite("training_data", &bob::visioner::param_t::m_train_data, "Training data")
    .def_readwrite("num_of_train_samples", &bob::visioner::param_t::m_train_samples, "Number of training samples")
    .def_readwrite("validation_data", &bob::visioner::param_t::m_valid_data, "Validation data")
    .def_readwrite("num_of_valid_samples", &bob::visioner::param_t::m_valid_samples, "Number of validation samples")
    .def_readwrite("feature_type", &bob::visioner::param_t::m_feature, "Feature type")
    .def_readwrite("feature_sharing", &bob::visioner::param_t::m_sharing, "Feature sharing")
    .def_readwrite("feature_projections", &bob::visioner::param_t::m_projections, "Coarse-to-fine feature projection")

    .def_readwrite("min_gt_overlap", &bob::visioner::param_t::m_min_gt_overlap, "Minimum overlapping with ground truth for positive samples")

    .def_readwrite("sliding_windows", &bob::visioner::param_t::m_ds, "Sliding windows")
    .def_readwrite("subwindow_labelling", &bob::visioner::param_t::m_tagger, "Labelling sub-windows")
    ;

  bp::enum_<bob::visioner::Sampler::SamplerType>("SamplerType")
    .value("Train", bob::visioner::Sampler::TrainSampler)
    .value("Validation", bob::visioner::Sampler::ValidSampler)
    ;

  bp::class_<bob::visioner::Sampler>("Sampler", "Object used for sampling uniformly, such that the same number of samples are obtained for distinct target values.", bp::init<bob::visioner::param_t, bob::visioner::Sampler::SamplerType>((bp::arg("param"), bp::arg("type")), "Default constructor with parameters and the type of sampler this sampler will be."))
    .def("__init__", make_constructor(&sampler_from_files, bp::default_call_policies(), (bp::arg("param"), bp::arg("type"), bp::arg("images"), bp::arg("ground_thruth"))), "Constructs a new sampler with parameters, a type and a list of images and (associated) ground-thruth information. Note that if you specify the list of images and ground-thruth inside the parameters object, that list will be read, but discarded in favor of the discrete list provided with the two input parameters.")
    .add_property("num_of_images", &bob::visioner::Sampler::n_images)
    .add_property("num_of_samples", &bob::visioner::Sampler::n_samples)
    .add_property("num_of_outputs", &bob::visioner::Sampler::n_outputs)
    .add_property("num_of_types", &bob::visioner::Sampler::n_types)
    .add_property("type", &bob::visioner::Sampler::getType, "This sampler's type")
    .def("load", &sampler_load, (bp::arg("self"), bp::arg("images"), bp::arg("ground_thruth")), "Resets the current contents of this sampler to use the image and (matching) ground-thruth files given. This method input lists or python iterables with the absolute or relative path of images and ground-thruth files you need to load.")
    ;

  bp::class_<bob::visioner::Model, boost::shared_ptr<bob::visioner::Model>, boost::noncopyable>("Model", "Multivariate model as a linear combination of LUTs. NB: The ::preprocess() must be called before ::get() and ::score() functions.", bp::no_init)
    .def("__init__", make_constructor(&bob::visioner::make_model, bp::default_call_policies(), (bp::arg("param"))), "Builds a new model from a parameter set.")
    .def("__init__", make_constructor(&model_from_path, bp::default_call_policies(), (bp::arg("path"))), "Builds a new model from a description sitting on a file.")
    .def("clone", &bob::visioner::Model::clone, (bp::arg("self")), "Clones the current model")
    .def("reset", &bob::visioner::Model::reset, (bp::arg("self"), bp::arg("param")), "Resets to new parameters")
    .def("project", &bob::visioner::Model::project, (bp::arg("self")), "Projects the selected features to a higher resolution")
    .def("save", (bool(bob::visioner::Model::*)(const std::string&) const)&bob::visioner::Model::save, (bp::arg("self"), bp::arg("path")), "Saves the model to a file")
    .def("get", &bob::visioner::Model::get, (bp::arg("self"), bp::arg("feature"), bp::arg("x"), bp::arg("y")), "Computes the value of the feature <f> at the (x, y) position")
    .add_property("num_of_features", &bob::visioner::Model::n_features)
    .add_property("num_of_fvalues", &bob::visioner::Model::n_fvalues)
    .add_property("num_of_outputs", &bob::visioner::Model::n_outputs)
    .def("num_of_luts", &bob::visioner::Model::n_luts, (bp::arg("self"), bp::arg("o")), "Number of LUTs")
    .def("describe", &bob::visioner::Model::describe, (bp::arg("self"), bp::arg("feature")), "Describes a feature")
    .def("train", &train_model, (bp::arg("self"), bp::arg("training_sampler"), bp::arg("validation_sampler")), "Trains the boosted classifier using training and validation samplers.")
    ;

  bp::scope().attr("LOSSES") = available_losses();
  bp::scope().attr("TAGGERS") = available_taggers();
  bp::scope().attr("MODELS") = available_models();
  bp::scope().attr("TRAINERS") = available_trainers();
  bp::scope().attr("OPTIMIZATIONS") = available_optimizations();
  bp::scope().attr("SHARINGS") = available_sharings();
}
