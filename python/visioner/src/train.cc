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

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include "core/python/ndarray.h"

#include "visioner/util/timer.h"
#include "visioner/model/mdecoder.h"
#include "visioner/model/sampler.h"

namespace bp = boost::python;
namespace tp = bob::python;

void bind_visioner_train() {
  bp::class_<bob::visioner::param_t>("param", "Various parameters useful for training boosted classifiers in the context of the Visioner", bp::init<bp::optional<bob::visioner::index_t, bob::visioner::index_t, const bob::visioner::string_t&, bob::visioner::scalar_t, const bob::visioner::string_t&, const bob::visioner::string_t&, bob::visioner::index_t, const bob::visioner::string_t&, const bob::visioner::string_t&, bob::visioner::index_t, bob::visioner::scalar_t, bob::visioner::index_t, const bob::visioner::string_t&> >((bp::arg("rows")=24, bp::arg("cols")=20, bp::arg("loss")="diag_log", bp::arg("loss_parameter")=0.0, bp::arg("optimization_type")="ept", bp::arg("training_model")="gboost", bp::arg("num_of_bootstraps")=3, bp::arg("feature_type")="elbp", bp::arg("feature_sharing")="shared", bp::arg("feature_projections")=0, bp::arg("min_gt_overlap")=0.8, bp::arg("sliding_windows")=2, bp::arg("subwindow_labelling")="object_type"), "Default constructor. Note: The seed, number of training and validation samples, as well as the maximum number of boosting rounds is hard-coded."))
    .def_readwrite("rows", &bob::visioner::param_t::m_rows, "Number of rows in pixels")
    .def_readwrite("cols", &bob::visioner::param_t::m_cols, "Number of columns in pixels")
    .def_readwrite("seed", &bob::visioner::param_t::m_seed, "Random seed used for sampling")

    .def_readwrite("loss", &bob::visioner::param_t::m_loss, "Loss")
    .def_readwrite("loss_parameter", &bob::visioner::param_t::m_loss_param, "Loss parameter")
    .def_readwrite("optimization_type", &bob::visioner::param_t::m_optimization, "Optimization type (expectation vs. variational)")
    .def_readwrite("training_model", &bob::visioner::param_t::m_trainer, "Training model")

    .def_readwrite("max_rounds", &bob::visioner::param_t::m_rounds, "Maximum boosting rounds")
    .def_readwrite("num_of_bootstraps", &bob::visioner::param_t::m_bootstraps, "Number of bootstrapping steps")

    .def_readwrite("num_of_train_samples", &bob::visioner::param_t::m_train_samples, "Number of training samples")
    .def_readwrite("num_of_valid_samples", &bob::visioner::param_t::m_valid_samples, "Number of validation samples")
    .def_readwrite("feature_type", &bob::visioner::param_t::m_feature, "Feature type")
    .def_readwrite("feature_sharing", &bob::visioner::param_t::m_sharing, "Feature sharing")
    .def_readwrite("feature_projections", &bob::visioner::param_t::m_projections, "Coarse-to-fine feature projection")

    .def_readwrite("min_gt_overlap", &bob::visioner::param_t::m_min_gt_overlap, "Minimum overlapping with ground truth for positive samples")

    .def_readwrite("sliding_windows", &bob::visioner::param_t::m_ds, "Sliding windows")
    .def_readwrite("subwindow_labelling", &bob::visioner::param_t::m_tagger, "Labelling sub-windows")
    ;
}
