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
#include <fftw3.h>

using namespace boost::python;

/**
 * FFTW3 support
 */
static tuple fftw3_version() {
  return make_tuple((const char*)fftw_version, 
                    (const char*)fftw_cc, 
                    (const char*)fftw_codelet_optim);
}

void bind_sp_version() {
  dict vdict;
  vdict["FFTW"] = fftw3_version();
  scope().attr("version") = vdict;
}
