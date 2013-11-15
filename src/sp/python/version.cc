/**
 * @file sp/python/version.cc
 * @date Tue Nov 29 14:11:41 2011 +0100
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Describes ways to retrieve version information about all dependent
 * packages.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#include <bob/config.h>

#include <boost/python.hpp>

using namespace boost::python;

void bind_sp_version() {
  dict vdict;
  scope().attr("version") = vdict;
}
