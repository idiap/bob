/**
 * @file bob/ip/DCTFeatures.h
 * @date Thu Apr 7 19:52:29 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file provides a class to extract DCT features as described in:
 *   "Polynomial Features for Robust Face Authentication",
 *   from C. Sanderson and K. Paliwal, in the proceedings of the
 *   IEEE International Conference on Image Processing 2002.
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

#ifndef BOB_IP_DCT_FEATURES_H
#define BOB_IP_DCT_FEATURES_H

#include "bob/core/cast.h"
#include "bob/core/array_copy.h"
#include "bob/ip/Exception.h"

#include <list>
#include <boost/shared_ptr.hpp>

#include "bob/ip/block.h"
#include "bob/sp/DCT2D.h"
#include "bob/ip/zigzag.h"


namespace bob {
/**
  * \ingroup libip_api
  * @{
  *
  */
  namespace ip {

  /**
    * @brief This class can be used to extract DCT features. This algorithm 
    *   is described in the following article:
    *   "Polynomial Features for Robust Face Authentication", 
    *   from C. Sanderson and K. Paliwal, in the proceedings of the 
    *   IEEE International Conference on Image Processing 2002.
    *   In addition, it support pre- and post-normalization (zero mean and 
    *   unit variance)
    */
  class DCTFeatures
  {
    public:

      /**
        * @brief Constructor: generates a DCTFeatures extractor
        * @param block_h height of the blocks
        * @param block_w width of the blocks
        * @param overlap_h overlap of the blocks along the y-axis
        * @param overlap_w overlap of the blocks along the x-axis
        * @param n_dct_coefs number of DCT coefficients to keep
        * @param norm_block Normalize the block to zero mean and
        *   unit variance before the DCT extraction. If the variance is zero
        *   (i.e. all the block values are the same), this will set all the 
        *   block values to zero before the DCT extraction. If both norm_block
        *   and norm_dct are set, the first DCT coefficient will always be 
        *   zero and could be removed afterwards.
        * @param norm_dct Normalize the DCT coefficients (across blocks) to 
        *   zero mean and unit variance after the DCT extraction. If the 
        *   variance is zero (i.e. a specific DCT coefficient is the same for 
        *   all the blocks), this will set these coefficients to a zero 
        *   constant. If both norm_block and norm_dct are set, the first DCT
        *   coefficient will always be zero and could be removed afterwards.
        */
      DCTFeatures( const size_t block_h, const size_t block_w, 
        const size_t overlap_h, const size_t overlap_w, 
        const size_t n_dct_coefs, const bool norm_block=false,
        const bool norm_dct=false): 
          m_dct2d(new bob::sp::DCT2D(block_h, block_w)),
          m_block_h(block_h), m_block_w(block_w), m_overlap_h(overlap_h), 
          m_overlap_w(overlap_w), m_n_dct_coefs(n_dct_coefs),
          m_norm_block(norm_block), m_norm_dct(norm_dct)
      {
        resetCache();
      }

      /**
        * @brief Copy constructor
        */
      DCTFeatures(const DCTFeatures& other):
        m_dct2d(new bob::sp::DCT2D(other.m_block_h, other.m_block_w)),
        m_block_h(other.m_block_h), m_block_w(other.m_block_w), 
        m_overlap_h(other.m_overlap_h), m_overlap_w(other.m_overlap_w),
        m_n_dct_coefs(other.m_n_dct_coefs), 
        m_norm_block(other.m_norm_block), m_norm_dct(other.m_norm_dct)
      {
        resetCache();
      }

      /**
        * @brief Destructor
        */
      virtual ~DCTFeatures() { }

      /**
        * @brief Assignment operator
        */
      DCTFeatures& operator=(const DCTFeatures& other);

      /**
        * @brief Equal to
        */
      bool operator==(const DCTFeatures& b) const;
      /**
        * @brief Not equal to
        */
      bool operator!=(const DCTFeatures& b) const; 

      /**
        * @brief Getters
        */
      size_t getBlockH() const { return m_block_h; }
      size_t getBlockW() const { return m_block_w; }
      size_t getOverlapH() const { return m_overlap_h; }
      size_t getOverlapW() const { return m_overlap_w; }
      size_t getNDctCoefs() const { return m_n_dct_coefs; }
      bool getNormalizeBlock() const { return m_norm_block; }
      bool getNormalizeDct() const { return m_norm_dct; }
 
      /**
        * @brief Setters
        */
      void setBlockH(const size_t block_h) 
      { m_block_h = block_h; m_dct2d->setHeight(block_h); 
        resetCacheBlock(); }
      void setBlockW(const size_t block_w) 
      { m_block_w = block_w; m_dct2d->setWidth(block_w); 
        resetCacheBlock(); }
      void setOverlapH(const size_t overlap_h) 
      { m_overlap_h = overlap_h; }
      void setOverlapW(const size_t overlap_w) 
      { m_overlap_w = overlap_w; }
      void setNDctCoefs(const size_t n_dct_coefs) 
      { m_n_dct_coefs = n_dct_coefs; resetCacheDct(); }
      void setNormalizeBlock(const bool norm_block)
      { m_norm_block = norm_block; }
      void setNormalizeDct(const bool norm_dct)
      { m_norm_dct = norm_dct; }
 
      /**
        * @brief Process a 2D blitz Array/Image by extracting DCT features.
        * @param src The 2D input blitz array
        * @param dst The 2D output array. The first dimension is for the block
        *   index, whereas the second one is for the dct index. The number of 
        *   expected blocks can be obtained using the getNBlocks() method
        */
      template <typename T> 
      void operator()(const blitz::Array<T,2>& src, blitz::Array<double,2>& dst) const;

      /**
        * @deprecated Please use the version with blitz::Array output. This
        *   version does not support block and/or dct normalization.
        * @brief Process a 2D blitz Array/Image by extracting DCT features.
        * @param src The 2D input blitz array
        * @param dst A container (with a push_back method such as an STL list)
        *   of 1D double blitz arrays.
        */
      template <typename T, typename U> 
      void operator()(const blitz::Array<T,2>& src, U& dst) const;

      /**
        * @deprecated Please use the version with 2D input blitz::Array. This
        *  version does not support block and/or dct normalization.
        * @brief Process a list of blocks by extracting DCT features.
        * @param src 3D input blitz array (list of 2D blocks)
        * @param dst 2D output blitz array
        */
      template <typename T>
      void operator()(const blitz::Array<T,3>& src, blitz::Array<double, 2>& dst) const;
      
      /**
        * @brief Function which returns the number of blocks when applying 
        *   the DCTFeatures extractor on a 2D blitz::array/image.
        *   The first dimension is the height (y-axis), whereas the second
        *   one is the width (x-axis).
        * @param src The input blitz array
        */
      template<typename T>
      size_t getNBlocks(const blitz::Array<T,2>& src) const;

    private:
      
      /**
        * Attributes
        */
      boost::shared_ptr<bob::sp::DCT2D> m_dct2d;
      size_t m_block_h;
      size_t m_block_w;
      size_t m_overlap_h;
      size_t m_overlap_w;
      size_t m_n_dct_coefs;
      bool m_norm_block;
      bool m_norm_dct;

      /**
        * Working arrays/variables in cache
        */
      void resetCache() const;
      void resetCacheBlock() const;
      void resetCacheDct() const;

      mutable blitz::Array<double,2> m_cache_block1;
      mutable blitz::Array<double,2> m_cache_block2;
      mutable blitz::Array<double,1> m_cache_dct1;
      mutable blitz::Array<double,1> m_cache_dct2;
  };

  // Declare template method full specialization
  template <>  
  void DCTFeatures::operator()<double>(const blitz::Array<double,2>& src, 
    blitz::Array<double,2>& dst) const;

  template <typename T>  
  void DCTFeatures::operator()(const blitz::Array<T,2>& src, 
    blitz::Array<double,2>& dst) const
  {   
    // Casts the input to double
    blitz::Array<double,2> src_d = bob::core::cast<double>(src);
    // Calls the specialized template function for double
    this->operator()(src_d, dst);
  }  

  template <typename T, typename U> 
  void DCTFeatures::operator()(const blitz::Array<T,2>& src, 
    U& dst) const
  { 
    // cast to double
    blitz::Array<double,2> double_version = bob::core::cast<double>(src);

    // get all the blocks
    std::list<blitz::Array<double,2> > blocks;
    blockReference(double_version, blocks, m_block_h, m_block_w, m_overlap_h, 
      m_overlap_w);
  
    /// dct extract each block
    for( std::list<blitz::Array<double,2> >::const_iterator it = blocks.begin(); 
      it != blocks.end(); ++it) 
    {
      // extract dct using operator()
      m_dct2d->operator()(bob::core::array::ccopy(*it), m_cache_block1);

      // extract the required number of coefficients using the zigzag pattern
      // Notice the allocation has push_back will call the copy constructor of
      // the blitz array, which does NOT reallocate/copy the data part!
      blitz::Array<double,1> dct_block_zigzag(m_n_dct_coefs);
      zigzag(m_cache_block1, dct_block_zigzag);
      
      // Push the resulting processed block in the container
      dst.push_back(dct_block_zigzag);
    }
  }

  template <typename T> 
  void DCTFeatures::operator()(const blitz::Array<T,3>& src, blitz::Array<double, 2>& dst) const
  { 
    // Cast to double
    blitz::Array<double,3> double_version = bob::core::cast<double>(src);

    bob::core::array::assertSameShape(src, blitz::TinyVector<int, 3>(src.extent(0), m_block_h, m_block_w));
    dst.resize(src.extent(0), m_n_dct_coefs);
    
    // Dct extract each block
    for(int i = 0; i < double_version.extent(0); ++i)
    {
      // Get the current block
      blitz::Array<double,2> dct_input = double_version(i, blitz::Range::all(), blitz::Range::all());

      // Extract dct using operator()
      m_dct2d->operator()(dct_input, m_cache_block1);

      // Extract the required number of coefficients using the zigzag pattern
      // and push it in the right dst row
      blitz::Array<double, 1> dst_row = dst(i, blitz::Range::all());
      zigzag(m_cache_block1, dst_row);
    }
  }
  
  template<typename T>
  size_t DCTFeatures::getNBlocks(const blitz::Array<T,2>& src) const
  {
    const blitz::TinyVector<int,3> res = getBlock3DOutputShape(src, m_block_h, 
      m_block_w, m_overlap_h, m_overlap_w); 
    return res(0);
  }

}}

#endif /* BOB_IP_DCT_FEATURES_H */
