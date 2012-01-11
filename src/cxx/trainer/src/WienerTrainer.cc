/**
 * @author Laurent El-Shafey <Laurent.El-Shafey@idiap.ch>
 * @date Thu 29 Sep 2011
 */

#include <vector>
#include <algorithm>

#include "trainer/WienerTrainer.h"
#include "io/Exception.h"
#include "core/array_type.h"
#include "core/cast.h"
#include "sp/FFT2D.h"

namespace io = bob::io;
namespace mach = bob::machine;
namespace train = bob::trainer;

train::WienerTrainer::WienerTrainer()
{
}

train::WienerTrainer::WienerTrainer(const train::WienerTrainer& other)
{
}

train::WienerTrainer::~WienerTrainer() 
{
}

train::WienerTrainer& train::WienerTrainer::operator=
(const train::WienerTrainer& other) 
{
  return *this;
}

void train::WienerTrainer::train(bob::machine::WienerMachine& machine, 
    const io::Arrayset& ar) const 
{
  // Checks for arrayset data type and shape once
  if (ar.getElementType() != bob::core::array::t_float64) {
    throw bob::io::TypeError(ar.getElementType(),
        bob::core::array::t_float64);
  }
  if (ar.getNDim() != 2) {
    throw bob::io::DimensionError(ar.getNDim(), 2);
  }

  // Data is checked now and conforms, just proceed w/o any further checks.
  size_t n_samples = ar.size();
  size_t height = ar.getShape()[0];
  size_t width = ar.getShape()[1];

  // FFT2D
  bob::sp::FFT2D fft2d(height, width);

  // Loads the data
  blitz::Array<double,3> data(height, width, n_samples);
  blitz::Array<std::complex<double>,2> sample_fft(height, width);
  blitz::Range all = blitz::Range::all();
  for (size_t i=0; i<n_samples; ++i) {
    blitz::Array<double,2> sample = ar.get<double,2>(i);
    blitz::Array<std::complex<double>,2> sample_c = bob::core::cast<std::complex<double> >(sample);
    fft2d(sample_c, sample_fft);
    data(all,all,i) = blitz::abs(sample_fft);
  }
  // Computes the mean of the training data
  blitz::Array<double,2> tmp(height,width);
  blitz::thirdIndex k;
  tmp = blitz::mean(data,k);
  // Removes the mean from the data
  for (size_t i=0; i<n_samples; ++i) {
    data(all,all,i) -= tmp;
  }
  // Computes power of 2 values
  data *= data;
  // Sums to get the variance
  tmp = blitz::sum(data,k) / n_samples;

  /**
   * sets the Wiener machine with the results:
   */
  machine.resize(height,width);
  machine.setPs(tmp);
}
