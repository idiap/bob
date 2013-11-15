/**
 * @file sp/python/main.cc
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 *
 * @brief Combines all modules to make up the complete bindings
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#include <bob/python/ndarray.h>

void bind_sp_version();
void bind_sp_extrapolate();
void bind_sp_dct_numpy();
void bind_sp_fft_numpy();
void bind_sp_convolution();
void bind_sp_convolution();
void bind_sp_quantization();

BOOST_PYTHON_MODULE(_sp) {
  boost::python::docstring_options docopt(true, true, false);
  bob::python::setup_python("bob signal processing classes and sub-classes");

  bind_sp_version();
  bind_sp_extrapolate();
  bind_sp_dct_numpy();
  bind_sp_fft_numpy();
  bind_sp_convolution();
  bind_sp_quantization();
}
