/**
 * @file machine/python/version.cc
 * @date Sat Dec 17 14:41:56 2011 +0100
 * @author André Anjos <andre.dos.anjos@gmail.com>
 *
 * @brief Describes ways to retrieve version information about all dependent
 * packages.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#include <bob/config.h>

#include <boost/python.hpp>
#include <boost/format.hpp>

#if WITH_LIBSVM
#include <svm.h>
#endif

using namespace boost::python;

/**
 * libsvm version
 */
static str get_libsvm_version() {
#if WITH_LIBSVM
  boost::format s("%d.%d.%d");
  s % (LIBSVM_VERSION / 100);
  s % ((LIBSVM_VERSION % 100) / 10);
  s % (LIBSVM_VERSION % 10);
  return str(s.str().c_str());
#else
  return str("unavailable");
#endif
}

void bind_machine_version() {
  dict vdict;
  vdict["libsvm"] = get_libsvm_version();
  scope().attr("version") = vdict;
}
