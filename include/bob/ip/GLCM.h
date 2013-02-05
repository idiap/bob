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
#include "bob/core/array_assert.h"
#include "bob/core/array_copy.h"
#include "bob/core/cast.h"
#include "bob/ip/Exception.h"
#include "bob/sp/interpolate.h"
#include "bob/ip/GLCMProp.h"

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
  class GLCM {

    public: //api

      /**
       * Complete constructor
       */
       
      /* 
      GLCM(const blitz::Array<int32_t, 1>& colOffset,
        const blitz::Array<int32_t, 1>& rowOffset,
        const bool autoMaxLevel = true,
        const bool autoMinLevel = true,
        const bool autoNumLevels = true,
        const bool symmetric=true,
        const bool normalized=true); 
      */
      GLCM();

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
      boost::shared_ptr<GLCM> clone() const;

      /**
      * Set the parameters for the quantization (scaling) of the gray-levels intensities
      */
      void setQuantizationParams(const int num_levels, const int min_level, const int max_level);
      
      /**
      * Set the parameters for the type of the output matrix (symmetric, normalized)
      */
      void setGLCMTypeParams(const bool symmetric, const bool normalized);
      
      /**
       * Get the required shape of the GLCM output blitz array, before calling
       * the operator() method.
       */
      template <typename T>  
        const blitz::TinyVector<int,3> getGLCMShape(const blitz::Array<T,2>& src) const;

      const blitz::TinyVector<int,3> getGLCMShape(const blitz::Array<uint8_t,2>& src) const
        { return getGLCMShape<uint8_t>(src); }
        
      const blitz::TinyVector<int,3> getGLCMShape(const blitz::Array<uint16_t,2>& src) const
        { return getGLCMShape<uint16_t>(src); }



      /**
       * Compute Gray-Level Co-occurences from a 2D blitz::Array, and save the resulting
       * GLCM matrix in the dst 3D blitz::Array.
       */
      template <typename T> 
        void operator()(const blitz::Array<T,2>& src, blitz::Array<double,3>& glcm) const;
        
      void 
        operator()(const blitz::Array<uint8_t,2>& src, blitz::Array<double,3>& glcm) const 
        { operator()<uint8_t>(src, glcm); }
      void 
        operator()(const blitz::Array<uint16_t,2>& src, blitz::Array<double,3>& glcm) const 
        { operator()<uint16_t>(src, glcm); }


      /**
      * Accessors
      */
      
      const blitz::Array<int32_t,2>&  getOffset() const
      { return m_offset; }
      const int getMaxLevel() const { return m_maxLevel; }
      const int getMinLevel() const { return m_minLevel; }
      const int getNumLevels() const { return m_numLevels; }
      const bool getSymmetric() const { return m_symmetric; }
      const bool getNormalized() const { return m_normalized; }
      const bool getRoundScaling() const { return m_roundScaling; }
      
      
      /**
      * Mutators
      */
      
      void setOffset(const blitz::Array<int32_t, 2>& offset)
      { m_offset.reference(bob::core::array::ccopy(offset)); }
            
      void setMaxLevel(const int maxLevel)
      { m_maxLevel = maxLevel; }

      void setMinLevel(const int minLevel)
      { m_minLevel = minLevel; }

      void setNumLevels(const int numLevels)
      { m_numLevels = numLevels; }
      
      void setSymmetric(const bool symmetric)
      { m_symmetric = symmetric; }

      void setNormalized(const bool normalized)
      { m_normalized = normalized; }
      
      void setRoundScaling(const bool roundScaling)
      { m_roundScaling = roundScaling; }

    protected:
    /**
    * Attributes
    */
    

    blitz::Array<int32_t,2> m_offset;
    int m_maxLevel;
    int m_minLevel;
    int m_numLevels;
    bool m_symmetric;
    bool m_normalized;
    bool m_roundScaling; // if true, the quantization (scaling) of the grey-scale values will be done as in Matlab, i.e. with rounding the discrete level. If false, the quantization (scaling) will be done uniformly
    
    /**
    * Methods
    */
    /**
    * Scales the input gray-scale value into a new value depending on the nuber of values, as well as the max and min values
    */
    const int scale_gray_value(int value, int max_value, int min_value, int num_values) const;
    
   };


template <typename T>
const blitz::TinyVector<int,3> bob::ip::GLCM::getGLCMShape(const blitz::Array<T,2>& src) const
{ 
  blitz::TinyVector<int,3> res;
  int num_levels, max_level, min_level;
  
  if (m_maxLevel == -1) // define the max level
    max_level = std::numeric_limits<T>::max();
  else
    max_level = m_maxLevel;

  if (m_minLevel == -1) // define the min level
    min_level = std::numeric_limits<T>::min();
  else
    min_level = m_minLevel;  
  
  if (m_numLevels == -1) // define the number of gray-scale levels
    num_levels = max_level - min_level + 1;
  else
    num_levels = m_numLevels; 
  
  res(0) = num_levels;
  res(1) = num_levels;
  res(2) = m_offset.extent(0); // the total number of offsets
  return res;
}




template <typename T>
void bob::ip::GLCM::operator()(const blitz::Array<T,2>& src, blitz::Array<double,3>& glcm) const
{
  // check if the size of the output matrix is as expected
  blitz::TinyVector<int,3> shape(getGLCMShape(src));
  bob::core::array::assertSameShape(glcm, shape);

  glcm=0;
  int num_levels, min_level, max_level;
  
  if (m_maxLevel == -1) // define the max level
    max_level = std::numeric_limits<T>::max();
  else
    max_level = m_maxLevel;

  if (m_minLevel == -1) // define the min level
    min_level = std::numeric_limits<T>::min();
  else
    min_level = m_minLevel;  
  
  if (m_numLevels == -1) // define the number of gray-scale levels
    num_levels = max_level - min_level + 1;
  else
    num_levels = m_numLevels; 
    
  
  for(int off_ind = 0; off_ind < m_offset.extent(0); ++off_ind) // loop over all the possible offsets
  {
      // loop over each pixel of the image
      for(int y = 0; y < src.extent(0); ++y)
      {
        for(int x = 0; x < src.extent(1); ++x)
        {
          int i_level = (int)(src(y,x)); // the grey level of the current pixel
          const int y1 = y + m_offset(off_ind, 1);
          const int x1 = x + m_offset(off_ind, 0);

          if(y1 >= 0 && y1 < src.extent(0) && x1 >= 0 && x1 < src.extent(1))
          {
            int j_level = (int)(src(y1, x1));
            glcm(scale_gray_value(i_level, max_level, min_level, num_levels), scale_gray_value(j_level, max_level, min_level, num_levels), off_ind) += 1; 
              
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

