/**
 * @file machine/cxx/WienerMachine.cc
 * @date Fri Sep 30 16:56:06 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implements a WienerMachine
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

#include <bob/core/array_copy.h>
#include <bob/core/cast.h>
#include <bob/machine/WienerMachine.h>
#include <bob/machine/Exception.h>
#include <bob/sp/FFT2D.h>
#include <complex>

bob::machine::WienerMachine::WienerMachine():
  m_Ps(0,0),
  m_variance_threshold(1e-8),
  m_Pn(0),
  m_W(0,0),
  m_fft(0,0),
  m_ifft(0,0),
  m_buffer1(0,0), m_buffer2(0,0)
{
}

bob::machine::WienerMachine::WienerMachine(const blitz::Array<double,2>& Ps, 
    const double Pn, const double variance_threshold):
  m_Ps(bob::core::array::ccopy(Ps)),
  m_variance_threshold(variance_threshold),
  m_Pn(Pn),
  m_W(m_Ps.extent(0),m_Ps.extent(1)),
  m_fft(m_Ps.extent(0),m_Ps.extent(1)),
  m_ifft(m_Ps.extent(0),m_Ps.extent(1)),
  m_buffer1(m_Ps.extent(0),m_Ps.extent(1)),
  m_buffer2(m_Ps.extent(0),m_Ps.extent(1))
{
  computeW();
}

bob::machine::WienerMachine::WienerMachine(const size_t height, 
    const size_t width, const double Pn, const double variance_threshold):
  m_Ps(height,width),
  m_variance_threshold(variance_threshold),
  m_Pn(Pn),
  m_W(height,width),
  m_fft(height,width),
  m_ifft(height,width),
  m_buffer1(0,0), m_buffer2(0,0)
{
  m_Ps = 1.;
  computeW();
}

bob::machine::WienerMachine::WienerMachine(const bob::machine::WienerMachine& other):
  m_Ps(bob::core::array::ccopy(other.m_Ps)),
  m_variance_threshold(other.m_variance_threshold),
  m_Pn(other.m_Pn),
  m_W(bob::core::array::ccopy(other.m_W)),
  m_fft(other.m_fft),
  m_ifft(other.m_ifft),
  m_buffer1(m_Ps.extent(0),m_Ps.extent(1)), 
  m_buffer2(m_Ps.extent(0),m_Ps.extent(1))
{
}

bob::machine::WienerMachine::WienerMachine(bob::io::HDF5File& config)
{
  load(config);
}

bob::machine::WienerMachine::~WienerMachine() {}

bob::machine::WienerMachine& bob::machine::WienerMachine::operator=
(const bob::machine::WienerMachine& other)
{
  if (this != &other)
  {
    m_Ps.reference(bob::core::array::ccopy(other.m_Ps));
    m_Pn = other.m_Pn;
    m_variance_threshold = other.m_variance_threshold;
    m_W.reference(bob::core::array::ccopy(other.m_W));
    m_fft.reset(m_Ps.extent(0),m_Ps.extent(1));
    m_ifft.reset(m_Ps.extent(0),m_Ps.extent(1));
    m_buffer1.resize(m_Ps.extent(0),m_Ps.extent(1));
    m_buffer2.resize(m_Ps.extent(0),m_Ps.extent(1));
  }
  return *this;
}

bool bob::machine::WienerMachine::operator==(const bob::machine::WienerMachine& b) const
{
  return bob::core::array::isEqual(m_Ps, b.m_Ps) &&
         m_variance_threshold == b.m_variance_threshold &&
         m_Pn == b.m_Pn &&
         bob::core::array::isEqual(m_W, b.m_W);
}

bool bob::machine::WienerMachine::operator!=(const bob::machine::WienerMachine& b) const 
{
  return !(this->operator==(b));
}

bool bob::machine::WienerMachine::is_similar_to(const bob::machine::WienerMachine& b,
  const double r_epsilon, const double a_epsilon) const
{
  return bob::core::array::isClose(m_Ps, b.m_Ps, r_epsilon, a_epsilon) &&
         bob::core::isClose(m_variance_threshold, b.m_variance_threshold, r_epsilon, a_epsilon) &&
         bob::core::isClose(m_Pn, b.m_Pn, r_epsilon, a_epsilon) &&
         bob::core::array::isClose(m_W, b.m_W, r_epsilon, a_epsilon);
}

void bob::machine::WienerMachine::load(bob::io::HDF5File& config)
{
  //reads all data directly into the member variables
  m_Ps.reference(config.readArray<double,2>("Ps"));
  m_Pn = config.read<double>("Pn");
  m_variance_threshold = config.read<double>("variance_threshold");
  m_W.reference(config.readArray<double,2>("W"));
  m_fft.reset(m_Ps.extent(0),m_Ps.extent(1));
  m_ifft.reset(m_Ps.extent(0),m_Ps.extent(1));
  m_buffer1.resize(m_Ps.extent(0),m_Ps.extent(1));
  m_buffer2.resize(m_Ps.extent(0),m_Ps.extent(1));
}

void bob::machine::WienerMachine::resize(const size_t height, 
  const size_t width)
{
  m_Ps.resizeAndPreserve(height,width);
  m_W.resizeAndPreserve(height,width);
  m_fft.reset(height,width);
  m_ifft.reset(height,width);
  m_buffer1.resizeAndPreserve(height,width);
  m_buffer2.resizeAndPreserve(height,width);
}

void bob::machine::WienerMachine::save(bob::io::HDF5File& config) const
{
  config.setArray("Ps", m_Ps);
  config.set("Pn", m_Pn);
  config.set("variance_threshold", m_variance_threshold);
  config.setArray("W", m_W);
}

void bob::machine::WienerMachine::computeW()
{
  // W = 1 / (1 + Pn / Ps_thresholded)
  m_W = 1. / (1. + m_Pn / m_Ps);
}


void bob::machine::WienerMachine::forward_(const blitz::Array<double,2>& input,
  blitz::Array<double,2>& output) const
{
  m_fft(bob::core::array::cast<std::complex<double> >(input), m_buffer1);
  m_buffer1 *= m_W;
  m_ifft(m_buffer1, m_buffer2);
  output = blitz::abs(m_buffer2);
}

void bob::machine::WienerMachine::forward(const blitz::Array<double,2>& input,
  blitz::Array<double,2>& output) const
{
  if (m_W.extent(0) != input.extent(0)) //checks input
    throw bob::machine::NInputsMismatch(m_W.extent(0),
        input.extent(0));
  if (m_W.extent(1) != input.extent(1)) //checks input
    throw bob::machine::NInputsMismatch(m_W.extent(1),
        input.extent(1));
  if (m_W.extent(0) != output.extent(0)) //checks output
    throw bob::machine::NOutputsMismatch(m_W.extent(0),
        output.extent(0));
  if (m_W.extent(1) != output.extent(1)) //checks output
    throw bob::machine::NOutputsMismatch(m_W.extent(1),
        output.extent(1));
  forward_(input, output);
}

void bob::machine::WienerMachine::setVarianceThreshold(
  const double variance_threshold)
{
  m_variance_threshold = variance_threshold;
  applyVarianceThreshold();
  computeW();
}

void bob::machine::WienerMachine::setPs(const blitz::Array<double,2>& Ps)
{
  if (m_Ps.extent(0) != Ps.extent(0))
    throw bob::machine::NInputsMismatch(m_Ps.extent(0), Ps.extent(0));
  if (m_Ps.extent(1) != Ps.extent(1))
    throw bob::machine::NInputsMismatch(m_Ps.extent(1), Ps.extent(0));
  m_Ps = bob::core::array::ccopy(Ps);
  computeW(); 
}

void bob::machine::WienerMachine::applyVarianceThreshold()
{
  m_Ps = blitz::where(m_Ps < m_variance_threshold, m_variance_threshold, m_Ps);
}
