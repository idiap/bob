/**
 * @file bob/sp/Quantization.h
 * @date Tue Feb  5 12:47:58 CET 2013
 * @author Ivana Chingovska <ivana.chingovska@idiap.ch>
 *
 * @brief Implement a blitz-based Quantization of 1D and 2D signals
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

#ifndef BOB_SP_QUANTIZATION_H
#define BOB_SP_QUANTIZATION_H

#include <blitz/array.h>
#include <bob/core/array_copy.h>
#include <bob/core/assert.h>
#include <limits>

namespace bob { namespace sp {

/**
 * @ingroup SP
 */
namespace quantization {
  /** 
   * @brief This enumeration defines different quantization methods
   */
  typedef enum {
    UNIFORM=0,
    UNIFORM_ROUNDING,
    USER_SPEC
  }   
  QuantizationType;
}

/**
 * @ingroup SP
 * @{
 */

/**
 * @brief This class implements a Quantization of signals.
 */
template <typename T> class Quantization
{
  public:
    /**
     * @brief Constructor: Initialize working array
     */
    Quantization();
    Quantization(const quantization::QuantizationType type, const int num_levels);
    Quantization(const quantization::QuantizationType type, const int num_levels, const T min_level, const T max_level);
    Quantization(const blitz::Array<T,1>& quant_thres); 

    /**
     * @brief Copy constructor
     */
    Quantization(const Quantization& other);

    /**
     * @brief Destructor
     */
    virtual ~Quantization();

    /**
     * @brief Assignment operator
     */
    Quantization& operator=(const Quantization& other);

            
    /**
     * @brief quantize a 1D array
     */ 
    void operator()(const blitz::Array<T,1>& src, blitz::Array<uint32_t,1>& res) const;
    blitz::Array<uint32_t,1> operator()(const blitz::Array<T,1>& src) const 
    {
      blitz::Array<uint32_t,1> res(src.extent(0));
      this->operator()(src, res);
      return res;
    }
 
    /**
     * @brief quantize a 2D array
     */
    void operator()(const blitz::Array<T,2>& src, blitz::Array<uint32_t,2>& res) const;
    blitz::Array<uint32_t,2> operator()(const blitz::Array<T,2>& src) const 
    {
      blitz::Array<uint32_t,2> res(src.extent(0), src.extent(1));
      this->operator()(src, res);
      return res;
    }
    
    /**
     * @brief determine the quantization level of one point
     */
    int quantization_level(const T src) const;
    

    /**
     * Accessors
     */   
    const blitz::Array<T,1>&  getThresholds() const { return m_thresholds; }
    const int getMaxLevel() const { return m_maxLevel; }
    const int getMinLevel() const { return m_minLevel; }
    const int getNumLevels() const { return m_numLevels; }
    const quantization::QuantizationType getType() const { return m_type; }


  protected:
    /// Attributes
    quantization::QuantizationType m_type;
    int m_numLevels;  
    int m_minLevel;
    int m_maxLevel;
    blitz::Array<T,1> m_thresholds;
    
    /// Methods
    /**
     * @brief Creates the table of thresholds, depending on the parameters of the class
     */
    void create_threshold_table();
};

/**
 * @}
 */
}}



template<typename T>
bob::sp::Quantization<T>::Quantization()
{
  m_thresholds = blitz::Array<T,1>();
  m_type = bob::sp::quantization::UNIFORM;
  m_maxLevel = std::numeric_limits<T>::max();
  m_minLevel = std::numeric_limits<T>::min();
  m_numLevels = m_maxLevel - m_minLevel + 1;
  create_threshold_table();
}

template<typename T>
bob::sp::Quantization<T>::Quantization(const blitz::Array<T,1>& quant_thres)
{
  m_thresholds.reference(bob::core::array::ccopy(quant_thres));
  m_type = bob::sp::quantization::USER_SPEC;
  m_maxLevel = std::numeric_limits<T>::max(); // the max_level is not known
  m_minLevel = m_thresholds(0);
  m_numLevels = m_thresholds.extent(0);
}

template<typename T>
bob::sp::Quantization<T>::Quantization(const bob::sp::quantization::QuantizationType type, const int num_levels):
  m_type(type), m_numLevels(num_levels)
{
  m_thresholds = blitz::Array<T,1>();
  m_maxLevel = std::numeric_limits<T>::max();
  m_minLevel = std::numeric_limits<T>::min();
  create_threshold_table();
}

template<typename T>
bob::sp::Quantization<T>::Quantization(const bob::sp::quantization::QuantizationType type, const int num_levels, const T min_level, const T max_level):
  m_type(type), m_numLevels(num_levels), m_minLevel(min_level), m_maxLevel(max_level)
{
  m_thresholds = blitz::Array<T,1>();
  create_threshold_table();
}


template<typename T>
bob::sp::Quantization<T>::Quantization(const bob::sp::Quantization<T>& other)
{
  m_thresholds.reference(bob::core::array::ccopy(other.getThresholds()));
  m_maxLevel = other.getMaxLevel();
  m_minLevel = other.getMinLevel();
  m_numLevels = other.getNumLevels();
  m_type = other.getType();
}

template<typename T>
bob::sp::Quantization<T>::~Quantization() { }

template<typename T>
bob::sp::Quantization<T>& bob::sp::Quantization<T>::operator=(const bob::sp::Quantization<T>& other) {
  if (this != &other)
  {
    m_thresholds.reference(bob::core::array::ccopy(other.getThresholds()));
    m_maxLevel = other.getMaxLevel();
    m_minLevel = other.getMinLevel();
    m_numLevels = other.getNumLevels();
    m_type = other.getType();
  }
  return *this;
}

template<typename T>
void bob::sp::Quantization<T>::operator()(const blitz::Array<T,2>& src, blitz::Array<uint32_t,2>& res) const
{ 
  bob::core::array::assertSameShape(src, res);
  
  for (int i=0; i < src.extent(0); ++i)
    for (int j=0; j < src.extent(1); ++j)
      res(i,j) = quantization_level(src(i,j));
}

template<typename T>
void bob::sp::Quantization<T>::operator()(const blitz::Array<T,1>& src, blitz::Array<uint32_t,1>& res) const
{ 
  bob::core::array::assertSameShape(src, res);
      
  for (int i=0; i < src.extent(0); ++i)
    res(i) = quantization_level(src(i));
}  

template<typename T>
int bob::sp::Quantization<T>::quantization_level(const T src) const
{   
  for (int i=0; i < m_thresholds.extent(0)-1; ++i)
  {
    if (src >= m_thresholds(i) && src < m_thresholds(i+1))
      return i;
  }
    
  if (src < m_thresholds(0))
    return 0;
      
  if (src >= m_thresholds(m_thresholds.extent(0)-1))
    return m_thresholds.extent(0)-1;  
      
  return 0;    
}


template<typename T>
void bob::sp::Quantization<T>::create_threshold_table()
{
  T range;
  switch(m_type)
  {
    case bob::sp::quantization::UNIFORM: // uniform quantization
    {
      m_thresholds.reference(blitz::Array<T,1>(m_numLevels));
      range = (m_maxLevel - m_minLevel + 2) / m_numLevels; //max_value-min_value+1 is the total number of values, +1 for correctly determining the range (in case of odd total number of values)
      m_thresholds(0) = m_minLevel;
      for (int i = 1; i < m_thresholds.extent(0); ++i)
        m_thresholds(i) = m_minLevel + i*range;
      break;
    }
            
    case bob::sp::quantization::UNIFORM_ROUNDING: // rounding quantization (as in Matlab)
    {
      m_thresholds.reference(blitz::Array<T,1>(m_numLevels));
      range = (m_maxLevel - m_minLevel + 2) / (m_numLevels - 1); //max_value-min_value+1 is the total number of values, +1 for correctly determining the range (in case of odd total number of values)
      m_thresholds(0) = m_minLevel;
      for (int i=1; i < m_thresholds.extent(0); ++i)
        m_thresholds(i) = m_minLevel + (i-1)*range + (range+1)/2 + 1;
      break;
    }
            
    case bob::sp::quantization::USER_SPEC: // quantization by pre-defined table of thresholds
    default:
      break;
   }
}   

#endif /* BOB_SP_QUANTIZATION_H */
