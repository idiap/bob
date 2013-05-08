/**
 * @file ap/cxx/FrameExtractor.cc
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

#include <bob/ap/FrameExtractor.h>
#include <bob/core/check.h>
#include <bob/core/assert.h>
#include <bob/core/cast.h>

bob::ap::FrameExtractor::FrameExtractor(const double sampling_frequency,
    const double win_length_ms, const double win_shift_ms):
  m_sampling_frequency(sampling_frequency), m_win_length_ms(win_length_ms),
  m_win_shift_ms(win_shift_ms)
{
  // Initialization
  initWinLength();
  initWinShift();
}

bob::ap::FrameExtractor::FrameExtractor(const FrameExtractor& other):
  m_sampling_frequency(other.m_sampling_frequency), 
  m_win_length_ms(other.m_win_length_ms), 
  m_win_shift_ms(other.m_win_shift_ms)
{
  // Initialization
  initWinLength();
  initWinShift();
}

bob::ap::FrameExtractor::~FrameExtractor()
{
}

bob::ap::FrameExtractor& bob::ap::FrameExtractor::operator=(const bob::ap::FrameExtractor& other)
{
  if (this != &other)
  {
    m_sampling_frequency = other.m_sampling_frequency;
    m_win_length_ms = other.m_win_length_ms;
    m_win_shift_ms = other.m_win_shift_ms;
    
    // Initialization
    initWinLength();
    initWinShift();
  }
  return *this;
}

bool bob::ap::FrameExtractor::operator==(const bob::ap::FrameExtractor& other) const
{
  return (m_sampling_frequency == other.m_sampling_frequency &&
      m_win_length_ms == other.m_win_length_ms &&
      m_win_shift_ms == other.m_win_shift_ms);
}

bool bob::ap::FrameExtractor::operator!=(const bob::ap::FrameExtractor& other) const
{
  return !(this->operator==(other));
}

void bob::ap::FrameExtractor::setSamplingFrequency(const double sampling_frequency)
{ 
  m_sampling_frequency = sampling_frequency;
  initWinLength();
  initWinShift();
}

void bob::ap::FrameExtractor::setWinLengthMs(const double win_length_ms)
{ 
  m_win_length_ms = win_length_ms;
  initWinLength(); 
}

void bob::ap::FrameExtractor::setWinShiftMs(const double win_shift_ms)
{ 
  m_win_shift_ms = win_shift_ms;
  initWinShift(); 
}

void bob::ap::FrameExtractor::initWinLength()
{ 
  m_win_length = (size_t)(m_sampling_frequency * m_win_length_ms / 1000);
  initWinSize();
}

void bob::ap::FrameExtractor::initWinShift()
{ 
  m_win_shift = (size_t)(m_sampling_frequency * m_win_shift_ms / 1000);
}

void bob::ap::FrameExtractor::initWinSize()
{
  m_win_size = (size_t)pow(2.0,ceil(log((double)m_win_length)/log(2)));
  m_cache_frame_d.resize(m_win_size);
}

void bob::ap::FrameExtractor::extractNormalizeFrame(const blitz::Array<double,1>& input,
  const size_t i, blitz::Array<double,1>& frame_d) const
{
  // Set padded frame to zero
  frame_d = 0.; 
  // Extract frame input vector
  blitz::Range rf(0,(int)m_win_length-1);
  blitz::Range ri(i*(int)m_win_shift,i*(int)m_win_shift+(int)m_win_length-1);
  frame_d(rf) = input(ri);
  // Subtract mean value
  frame_d -= blitz::mean(frame_d);
}


blitz::TinyVector<int,2> 
bob::ap::FrameExtractor::getShape(const size_t input_size) const
{
  // Res will contain the number of frames x the dimension of the feature vector
  blitz::TinyVector<int,2> res;

  // 1. Number of frames
  res(0) = 1+((input_size-m_win_length)/m_win_shift);

  // 2. Dimension of the feature vector
  res(1) = m_win_length;

  return res;
}

blitz::TinyVector<int,2>
bob::ap::FrameExtractor::getShape(const blitz::Array<double,1>& input) const
{
  return getShape(input.extent(0));
}

