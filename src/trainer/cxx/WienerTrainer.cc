/**
 * @file trainer/cxx/WienerTrainer.cc
 * @date Fri Sep 30 16:58:42 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
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

#include <bob/trainer/WienerTrainer.h>
#include <bob/machine/Exception.h>
#include <bob/core/cast.h>
#include <bob/sp/FFT2D.h>
#include <complex>

bob::trainer::WienerTrainer::WienerTrainer()
{
}

bob::trainer::WienerTrainer::WienerTrainer(const bob::trainer::WienerTrainer& other)
{
}

bob::trainer::WienerTrainer::~WienerTrainer() 
{
}

bob::trainer::WienerTrainer& bob::trainer::WienerTrainer::operator=
(const bob::trainer::WienerTrainer& other) 
{
  return *this;
}

bool bob::trainer::WienerTrainer::operator==
  (const bob::trainer::WienerTrainer& other) const
{
  return true;
}

bool bob::trainer::WienerTrainer::operator!=
  (const bob::trainer::WienerTrainer& other) const
{
  return !(this->operator==(other));
}

bool bob::trainer::WienerTrainer::is_similar_to
  (const bob::trainer::WienerTrainer& other, const double r_epsilon,
   const double a_epsilon) const
{
  return true;
}

void bob::trainer::WienerTrainer::train(bob::machine::WienerMachine& machine, 
    const blitz::Array<double,3>& ar)
{
  // Data is checked now and conforms, just proceed w/o any further checks.
  const size_t n_samples = ar.extent(0);
  const size_t height = ar.extent(1);
  const size_t width = ar.extent(2);
  // machine dimensions
  const size_t height_m = machine.getHeight();
  const size_t width_m = machine.getWidth();

  // Checks that the dimensions are matching
  if (height != height_m)
    throw bob::machine::NInputsMismatch(height, height_m);
  if (width != width_m)
    throw bob::machine::NInputsMismatch(width, width_m);

  // FFT2D
  bob::sp::FFT2D fft2d(height, width);

  // Loads the data
  blitz::Array<double,3> data(height, width, n_samples);
  blitz::Array<std::complex<double>,2> sample_fft(height, width);
  blitz::Range all = blitz::Range::all();
  for (size_t i=0; i<n_samples; ++i) {
    blitz::Array<double,2> sample = ar(i,all,all);
    blitz::Array<std::complex<double>,2> sample_c = bob::core::array::cast<std::complex<double> >(sample);
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

  // sets the Wiener machine with the results:
  machine.setPs(tmp);
}
