/**
 * @file cxx/machine/src/WienerMachine.cc
 * @date Fri Sep 30 16:56:06 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implements a WienerMachine
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

#include "core/array_copy.h"
#include "core/cast.h"
#include "io/Arrayset.h"
#include "machine/WienerMachine.h"
#include "machine/Exception.h"
#include <complex>

namespace mach = bob::machine;

mach::WienerMachine::WienerMachine(const blitz::Array<double,2>& Ps, const double Pn,
    const double variance_threshold):
  m_Ps(bob::core::array::ccopy(Ps)),
  m_variance_threshold(variance_threshold),
  m_Pn(Pn),
  m_fft(new bob::sp::FFT2D(m_Ps.extent(0),m_Ps.extent(1))),
  m_ifft(new bob::sp::IFFT2D(m_Ps.extent(0),m_Ps.extent(1))),
  m_buffer1(m_Ps.extent(0),m_Ps.extent(1)),
  m_buffer2(m_Ps.extent(0),m_Ps.extent(1))
{
  m_W.resize(m_Ps.shape());
  computeW();
}

mach::WienerMachine::WienerMachine():
  m_Ps(0,0),
  m_variance_threshold(1e-8),
  m_Pn(0),
  m_W(0,0),
  m_fft(boost::shared_ptr<bob::sp::FFT2D>()),
  m_ifft(boost::shared_ptr<bob::sp::IFFT2D>()),
  m_buffer1(0,0), m_buffer2(0,0)
{
}

mach::WienerMachine::WienerMachine(size_t height, size_t width, const double Pn,
    const double variance_threshold):
  m_Ps(height,width),
  m_variance_threshold(variance_threshold),
  m_Pn(Pn),
  m_W(height,width),
  m_fft(new bob::sp::FFT2D(height,width)),
  m_ifft(new bob::sp::IFFT2D(height,width)),
  m_buffer1(0,0), m_buffer2(0,0)
{
  m_Ps = 0.;
  m_W = 0.;
}

mach::WienerMachine::WienerMachine(const mach::WienerMachine& other):
  m_Ps(bob::core::array::ccopy(other.m_Ps)),
  m_variance_threshold(other.m_variance_threshold),
  m_Pn(other.m_Pn),
  m_W(bob::core::array::ccopy(other.m_W)),
  m_fft(new bob::sp::FFT2D(m_Ps.extent(0),m_Ps.extent(1))),
  m_ifft(new bob::sp::IFFT2D(m_Ps.extent(0),m_Ps.extent(1))),
  m_buffer1(m_Ps.extent(0),m_Ps.extent(1)), 
  m_buffer2(m_Ps.extent(0),m_Ps.extent(1))
{
}

mach::WienerMachine::WienerMachine (bob::io::HDF5File& config) {
  load(config);
}

mach::WienerMachine::~WienerMachine() {}

mach::WienerMachine& mach::WienerMachine::operator=
(const mach::WienerMachine& other) {
  m_Ps.reference(bob::core::array::ccopy(other.m_Ps));
  m_Pn = other.m_Pn;
  m_variance_threshold = other.m_variance_threshold;
  m_W.reference(bob::core::array::ccopy(other.m_W));
  m_fft.reset(new bob::sp::FFT2D(m_Ps.extent(0),m_Ps.extent(1)));
  m_ifft.reset(new bob::sp::IFFT2D(m_Ps.extent(0),m_Ps.extent(1)));
  m_buffer1.resize(m_Ps.extent(0),m_Ps.extent(1));
  m_buffer2.resize(m_Ps.extent(0),m_Ps.extent(1));
  return *this;
}

void mach::WienerMachine::load (bob::io::HDF5File& config) {
  //reads all data directly into the member variables
  m_Ps.reference(config.readArray<double,2>("Ps"));
  m_Pn = config.read<double>("Pn");
  m_variance_threshold = config.read<double>("variance_threshold");
  m_W.reference(config.readArray<double,2>("W"));
  m_fft.reset(new bob::sp::FFT2D(m_Ps.extent(0),m_Ps.extent(1)));
  m_ifft.reset(new bob::sp::IFFT2D(m_Ps.extent(0),m_Ps.extent(1)));
  m_buffer1.resize(m_Ps.extent(0),m_Ps.extent(1));
  m_buffer2.resize(m_Ps.extent(0),m_Ps.extent(1));
}

void mach::WienerMachine::resize (size_t height, size_t width) {
  m_Ps.resizeAndPreserve(height,width);
  m_W.resizeAndPreserve(height,width);
  m_fft.reset(new bob::sp::FFT2D(height,width));
  m_ifft.reset(new bob::sp::IFFT2D(height,width));
  m_buffer1.resizeAndPreserve(height,width);
  m_buffer2.resizeAndPreserve(height,width);
}

void mach::WienerMachine::save (bob::io::HDF5File& config) const {
  config.setArray("Ps", m_Ps);
  config.set("Pn", m_Pn);
  config.set("variance_threshold", m_variance_threshold);
  config.setArray("W", m_W);
}

void mach::WienerMachine::computeW () {
  m_W = m_Ps;
  // Apply variance flooring threshold
  blitz::Array<bool,2> isTooSmall(m_W.shape());
  isTooSmall = m_Ps < m_variance_threshold;
  m_W += (m_variance_threshold - m_W) * isTooSmall; // W = Pn_thresholded
  // W = 1 / (1 + Pn / Ps_thresholded)
  m_W = 1. / (1. + m_Pn / m_W);
}


void mach::WienerMachine::forward_
(const blitz::Array<double,2>& input, blitz::Array<double,2>& output) const {
  m_fft->operator()(bob::core::cast<std::complex<double> >(input), m_buffer1);
  m_buffer1 *= m_W;
  m_ifft->operator()(m_buffer1, m_buffer2);
  output = blitz::abs(m_buffer2);
}

void mach::WienerMachine::forward
(const blitz::Array<double,2>& input, blitz::Array<double,2>& output) const {
  if (m_W.extent(0) != input.extent(0)) //checks input
    throw mach::NInputsMismatch(m_W.extent(0),
        input.extent(0));
  if (m_W.extent(1) != input.extent(1)) //checks input
    throw mach::NInputsMismatch(m_W.extent(1),
        input.extent(1));
  if (m_W.extent(0) != output.extent(0)) //checks output
    throw mach::NOutputsMismatch(m_W.extent(0),
        output.extent(0));
  if (m_W.extent(1) != output.extent(1)) //checks output
    throw mach::NOutputsMismatch(m_W.extent(1),
        output.extent(1));
  forward_(input, output);
}

void mach::WienerMachine::setPs(const blitz::Array<double,2>& Ps) { 
  if (m_Ps.extent(0) != Ps.extent(0)) {
    throw mach::NInputsMismatch(m_Ps.extent(0), Ps.extent(0));
  }
  if (m_Ps.extent(1) != Ps.extent(1)) {
    throw mach::NInputsMismatch(m_Ps.extent(1), Ps.extent(0));
  }
  m_Ps = bob::core::array::ccopy(Ps);
  computeW(); 
}
