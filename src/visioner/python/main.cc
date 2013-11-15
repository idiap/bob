/**
 * @file visioner/python/main.cc
 * @date Thu Jul 21 13:13:06 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Combines all modules to make up the complete bindings
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#include "bob/python/ndarray.h"

void bind_visioner_version();
void bind_visioner_localize();
void bind_visioner_train();

BOOST_PYTHON_MODULE(_visioner) {
  boost::python::docstring_options docopt(true, true, false);
  bob::python::setup_python("Face detection, keypoint localization and pose estimation using Boosting and LBP-like features (Visioner)");

  bind_visioner_version();
  bind_visioner_localize();
  bind_visioner_train();
}
