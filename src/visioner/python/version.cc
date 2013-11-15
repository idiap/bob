/**
 * @file visioner/python/version.cc
 * @date Tue Nov 29 14:11:41 2011 +0100
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Describes ways to retrieve version information about all dependent
 * packages.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#include "bob/config.h"

#include <boost/python.hpp>
#if WITH_QT4
#include <qglobal.h>
#endif

using namespace boost::python;

/**
 * Qt4 version, if available
 */
static tuple qt4_version() {
#if WITH_QT4
  return make_tuple(str(QT_VERSION_STR), str(QT_PACKAGEDATE_STR));
#else
  return make_tuple(str("unavailable"), str("unknown"));
#endif
}

void bind_visioner_version() {
  dict vdict;
  vdict["Qt4"] = qt4_version();
  scope().attr("version") = vdict;
}
