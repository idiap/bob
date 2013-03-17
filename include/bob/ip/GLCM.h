/**
 * @file bob/ip/GLCM.h
 * @date Tue Jan 22 12:31:59 CET 2013
 * @author Ivana Chingovska <ivana.chingovska@idiap.ch>
 *
 * This file defines a function to compute the Grey Level Co-occurence Matrix (GLCM)
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

#ifndef BOB_IP_GLCM_H
#define BOB_IP_GLCM_H

#include <math.h>
#include <iostream>
#include <blitz/array.h>
#include <algorithm>
#include <boost/shared_ptr.hpp>
#include "bob/core/assert.h"
#include "bob/core/array_copy.h"
#include "bob/core/cast.h"
#include "bob/ip/Exception.h"
#include "bob/sp/Quantization.h"

namespace bob { namespace ip {

  /**
   * This class allows to extract Grey-Level Co-occurence Matrix (GLCM). For more information, please refer to the
   * following article: "Textural Features for Image calssification", from R. M. Haralick, K. Shanmugam, I. Dinstein
   * in the IEEE Transactions on Systems, Man and Cybernetics, vol.SMC-3, No. 6, p. 610-621.
   *
   * A thorough tutorial about GLCM and the textural (so-called Haralick) properties that can be derived from it, can be found at: 
   * http://www.fp.ucalgary.ca/mhallbey/tutorial.htm
   *
   * List of references:
   * [1] R. M. Haralick, K. Shanmugam, I. Dinstein; "Textural Features for Image calssification",
   * in IEEE Transactions on Systems, Man and Cybernetics, vol.SMC-3, No. 6, p. 610-621.
   * [2] http://www.mathworks.ch/ch/help/images/ref/graycomatrix.html
   */
  template <typename T> 
  class GLCM {

    public: //api

      /**
       * Complete constructor
       */
       
      GLCM();
      GLCM(const int num_levels);
      GLCM(const int num_levels, const T min_level, const T max_level);
      GLCM(const blitz::Array<T,1>& quant_thres); 

      /**
       * Copy constructor
       */
      GLCM(const GLCM& other);

      /**
       * Destructor
       */
      virtual ~GLCM();

      /**
       * Assignment
       */
      GLCM& operator= (const GLCM& other);

      /**
       * Clone self into a boost::shared_ptr<GLCM>
       */
      //boost::shared_ptr<GLCM> clone() const;
      
      
      
      /**
       * Get the required shape of the GLCM output blitz array, before calling
       * the operator() method.
       */
      const blitz::TinyVector<int,3> getGLCMShape() const;


      /**
       * Compute Gray-Level Co-occurences from a 2D blitz::Array, and save the resulting
       * GLCM matrix in the dst 3D blitz::Array.
       */
      void operator()(const blitz::Array<T,2>& src, blitz::Array<double,3>& glcm) const;

      /**
      * Accessors
      */
      
      const blitz::Array<int32_t,2>&  getOffset() const
      { return m_offset; }
      const int getMaxLevel() const { return m_quantization.getMaxLevel(); }
      const int getMinLevel() const { return m_quantization.getMinLevel(); }
      const int getNumLevels() const { return m_quantization.getNumLevels(); }
      const bool getSymmetric() const { return m_symmetric; }
      const bool getNormalized() const { return m_normalized; }
      const bob::sp::Quantization<T> getQuantization() const { return m_quantization; }
      const blitz::Array<T,1>&  getQuantizationTable() const{ return m_quantization.getThresholds(); }
      
      
      /**
      * Mutators
      */
      
      void setOffset(const blitz::Array<int32_t, 2>& offset)
      { m_offset.reference(bob::core::array::ccopy(offset)); }
      
      void setSymmetric(const bool symmetric)
      { m_symmetric = symmetric; }

      void setNormalized(const bool normalized)
      { m_normalized = normalized; }
      
    protected:
    /**
    * Attributes
    */
    

    blitz::Array<int32_t,2> m_offset;
    bob::sp::Quantization<T> m_quantization;
    bool m_symmetric;
    bool m_normalized;
    
    
   };


template <typename T>
bob::ip::GLCM<T>::GLCM()
{
  m_offset.reference(blitz::Array<int32_t,2>(1,2));
  m_offset = 1, 0; // this is the default offset
  m_symmetric = false;
  m_normalized = false;
  m_quantization = bob::sp::Quantization<T>();
}

template <typename T>
bob::ip::GLCM<T>::GLCM(int num_levels)
{
  m_offset.reference(blitz::Array<int32_t,2>(1,2));
  m_offset = 1, 0; // this is the default offset
  m_symmetric = false;
  m_normalized = false;
  m_quantization = bob::sp::Quantization<T>(bob::sp::quantization::UNIFORM, num_levels);
}

template <typename T>
bob::ip::GLCM<T>::GLCM(int num_levels, T min_level, T max_level)
{
  m_offset.reference(blitz::Array<int32_t,2>(1,2));
  m_offset = 1, 0; // this is the default offset
  m_symmetric = false;
  m_normalized = false;
  m_quantization = bob::sp::Quantization<T>(bob::sp::quantization::UNIFORM, num_levels, min_level, max_level);
}

template <typename T>
bob::ip::GLCM<T>::GLCM(const blitz::Array<T,1>& quant_thres)
{
  m_offset.reference(blitz::Array<int32_t,2>(1,2));
  m_offset = 1, 0; // this is the default offset
  m_symmetric = false;
  m_normalized = false;
  m_quantization = bob::sp::Quantization<T>(quant_thres);
}

template <typename T>
bob::ip::GLCM<T>::GLCM(const bob::ip::GLCM<T>& other)
{
  m_offset.reference(bob::core::array::ccopy(other.getOffset()));
  m_symmetric = other.getSymmetric();
  m_normalized = other.getNormalized();
  m_quantization = other.getQuantization();
}

template <typename T>
bob::ip::GLCM<T>::~GLCM() { }

template <typename T>
bob::ip::GLCM<T>& bob::ip::GLCM<T>::operator=(const bob::ip::GLCM<T>& other) {
  if(this != &other)
  {
    m_offset.reference(bob::core::array::ccopy(other.getOffset()));
    m_symmetric = other.getSymmetric();
    m_normalized = other.getNormalized();
    m_quantization = other.getQuantization();
  }
  return *this;
}

/*
template <typename T>
boost::shared_ptr<bob::ip::GLCM<T>> bob::ip::GLCM<T>::clone() const {
  return boost::make_shared<bob::ip::GLCM>(*this);
}
*/


template <typename T>
const blitz::TinyVector<int,3> bob::ip::GLCM<T>::getGLCMShape() const
{ 
  blitz::TinyVector<int,3> res;
  
  
  res(0) = m_quantization.getNumLevels();
  res(1) = m_quantization.getNumLevels();
  res(2) = m_offset.extent(0); // the total number of offsets
  return res;
}




template <typename T>
void bob::ip::GLCM<T>::operator()(const blitz::Array<T,2>& src, blitz::Array<double,3>& glcm) const
{
  // check if the size of the output matrix is as expected
  blitz::TinyVector<int,3> shape(getGLCMShape());
  bob::core::array::assertSameShape(glcm, shape);

  glcm=0;
  
  blitz::Array<uint32_t,2> src_quant = m_quantization(src);
  
  for(int off_ind = 0; off_ind < m_offset.extent(0); ++off_ind) // loop over all the possible offsets
  {
      // loop over each pixel of the image
      for(int y = 0; y < src_quant.extent(0); ++y)
      {
        for(int x = 0; x < src_quant.extent(1); ++x)
        {
          int i_level = (int)(src_quant(y,x)); // the grey level of the current pixel
          const int y1 = y + m_offset(off_ind, 1);
          const int x1 = x + m_offset(off_ind, 0);

          if(y1 >= 0 && y1 < src_quant.extent(0) && x1 >= 0 && x1 < src_quant.extent(1))
          {
            int j_level = (int)(src_quant(y1, x1));
            glcm(i_level, j_level, off_ind) += 1; 
              
          }
        }
      }   
            
    }
  
  if(m_symmetric) // make the matrix symmetric
  {
    blitz::Array<double,3> temp = glcm.copy();
    temp.transposeSelf(1,0,2);
    glcm += temp;
  }

  if (m_normalized) // normalize the output image
  {
    blitz::firstIndex i;
    blitz::secondIndex j;
    blitz::thirdIndex k;
    blitz::Array<double, 2> summations_temp(blitz::sum(glcm(i, k, j), k));
    blitz::Array<double, 1> summations(blitz::sum(summations_temp(j,i), j));
    glcm /= summations(k);
    //std::cout << "glcm after normalization: " << glcm(blitz::Range(0,1),blitz::Range(0,1),blitz::Range(0,1)) << std::endl;    
  }
  
}        

}}

#endif /* BOB_IP_GLCM_H */  

