/**
 * @file ap/cxx/Energy.cc
 * @date Wed Jan 11:09:30 2013 +0200
 * @author Elie Khoury <Elie.Khoury@idiap.ch>
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

#include <bob/ap/Energy.h>
#include <bob/core/array_assert.h>

bob::ap::Energy::Energy(const double sampling_frequency, const double win_length_ms,
    const double win_shift_ms):
  bob::ap::FrameExtractor(sampling_frequency, win_length_ms, win_shift_ms),
  m_energy_floor(1.)
{
  // Initializes logarithm of flooring values
  m_log_energy_floor = log(m_energy_floor);
}

bob::ap::Energy::Energy(const bob::ap::Energy& other):
  bob::ap::FrameExtractor(other), m_energy_floor(1.)
{
  // Initializes logarithm of flooring values
  m_log_energy_floor = log(m_energy_floor);
}

bob::ap::Energy& bob::ap::Energy::operator=(const bob::ap::Energy& other)
{
  if (this != &other)
  {
    bob::ap::FrameExtractor::operator=(other);
    m_energy_floor = other.m_energy_floor;
    // Initializes logarithm of flooring values
    m_log_energy_floor = log(m_energy_floor);
  }
  return *this;
}

bool bob::ap::Energy::operator==(const bob::ap::Energy& other) const
{
  return (bob::ap::FrameExtractor::operator==(other) && 
          m_energy_floor == other.m_energy_floor);
}

bool bob::ap::Energy::operator!=(const bob::ap::Energy& other) const
{
  return !(bob::ap::Energy::operator==(other));
}

bob::ap::Energy::~Energy()
{
}


blitz::TinyVector<int,2>
bob::ap::Energy::getShape(const size_t input_size) const
{
  blitz::TinyVector<int,2> res = bob::ap::FrameExtractor::getShape(input_size);
  res(1) = 1;
  return res;
}

blitz::TinyVector<int,2>
bob::ap::Energy::getShape(const blitz::Array<double,1>& input) const
{
  return getShape(input.extent(0));
}

void bob::ap::Energy::operator()(const blitz::Array<double,1>& input, 
  blitz::Array<double,1>& energy_array)
{
  // Get expected dimensionality of output array
  int n_frames = bob::ap::Energy::getShape(input)(0);
  // Check dimensionality of output array
  bob::core::array::assertSameDimensionLength(energy_array.extent(0), n_frames);

  for (int i=0; i<n_frames; ++i)
  {
    // Extract and normalize frame
    extractNormalizeFrame(input, i, m_cache_frame_d);

    // Update output with logEnergy
    energy_array(i) = logEnergy(m_cache_frame_d);
  }
}

double bob::ap::Energy::logEnergy(blitz::Array<double,1> &data) const
{
  blitz::Array<double,1> data_p(data(blitz::Range(0,(int)m_win_length-1)));
  double gain = blitz::sum(blitz::pow2(data_p));
  return (gain < m_energy_floor ? m_log_energy_floor : log(gain)); 
}

