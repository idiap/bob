/**
 * @file ip/cxx/GLCM.cc
 * @date Tue Jan 22 15:59:37 CET 2013
 * @author Ivana Chingovska <ivana.chingovska@idiap.ch> 
 *
 * @brief GLCM implementation
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

#include "bob/ip/GLCM.h"
#include "bob/core/array_copy.h"
#include <boost/make_shared.hpp>


bob::ip::GLCM::GLCM()
{
  m_offset.reference(blitz::Array<int32_t,2>(1,2));
  m_offset = 1, 0; // this is the default offset
  m_symmetric = false;
  m_normalized = false;
  m_roundScaling = false;
  m_maxLevel = -1;
  m_minLevel = -1;
  m_numLevels = -1;
}


bob::ip::GLCM::GLCM(const bob::ip::GLCM& other)
{
  m_offset.reference(bob::core::array::ccopy(other.getOffset()));
  m_maxLevel = other.getMaxLevel();
  m_minLevel = other.getMinLevel();
  m_numLevels = other.getNumLevels();
  m_symmetric = other.getSymmetric();
  m_normalized = other.getNormalized();
  m_roundScaling = other.getRoundScaling();
}


bob::ip::GLCM::~GLCM() { }

bob::ip::GLCM& bob::ip::GLCM::operator=(const bob::ip::GLCM& other) {
  if(this != &other)
  {
    m_offset.reference(bob::core::array::ccopy(other.getOffset()));
    m_maxLevel = other.getMaxLevel();
    m_minLevel = other.getMinLevel();
    m_numLevels = other.getNumLevels();
    m_symmetric = other.getSymmetric();
    m_normalized = other.getNormalized();
    m_roundScaling = other.getRoundScaling();
  }
  return *this;
}

void bob::ip::GLCM::setQuantizationParams(const int num_levels, const int min_level, const int max_level)
{
  m_numLevels = num_levels;
  m_minLevel = min_level;
  m_maxLevel = max_level;
}  

void bob::ip::GLCM::setGLCMTypeParams(const bool symmetric, const bool normalized)
{
  m_symmetric = symmetric;
  m_normalized = normalized;
}


const int bob::ip::GLCM::scale_gray_value(int value, int max_value, int min_value, int num_levels) const
{
  if (value > max_value)
    return num_levels - 1;
  else
    if (value < min_value)
      return 0;
    else
      if (num_levels == max_value - min_value + 1) // later i need to remove this and merge with the next possibility
        return value;
      else
      {  
        if (!m_roundScaling)
        {
          int range = (max_value - min_value + 2) / num_levels; //max_value-min_value+1 is the total number of values, +1 for correctly determining the range (in case of odd total number of values)
          return (value - min_value) / range; // do the uniform quantization
        }  
        else // this is currently not working exactly like matlab
        {
          int range = (max_value - min_value + 2) / (num_levels - 1); //max_value-min_value+1 is the total number of values, +1 for correctly determining the range (in case of odd total number of values)
          return (value - min_value + (range+1)/2) / range; // compatibility with Matlab method graycomatrix(). +1 added to range in case of a odd range
        }  
      }  
}

boost::shared_ptr<bob::ip::GLCM> bob::ip::GLCM::clone() const {
  return boost::make_shared<bob::ip::GLCM>(*this);
}







  
