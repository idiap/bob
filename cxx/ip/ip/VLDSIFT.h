/**
 * @file cxx/ip/ip/VLDSIFT.h
 * @date Mon Jan 23 20:46:07 2012 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file defines classes to compute SIFT features using VLFeat
 *
 * Copyright (C) 2011-2012 Idiap Reasearch Institute, Martigny, Switzerland
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

#ifndef BOB_IP_VLDSIFT_H
#define BOB_IP_VLDSIFT_H

#include <blitz/array.h>
#include <vl/dsift.h>

namespace bob {
/**
 * \ingroup libip_api
 * @{
 *
 */
  namespace ip {

    /**
      * @brief This class allows the computation of Dense SIFT features.
      *   The computation is done using the VLFeat library
      *   For more information, please refer to the following article:
      *     "Distinctive Image Features from Scale-Invariant Keypoints", 
      *     from D.G. Lowe,
      *     International Journal of Computer Vision, 60, 2, pp. 91-110, 2004
      */
    class VLDSIFT
    {
      public:
        /**
          * @brief Constructor
          * @param height Input/image height
          * @param width Input/image width
          * @param step The x- and y-step for generating the grid of keypoins
          * @param block_size The x and y- size of a unit block
          */
        VLDSIFT(const int height, const int width, const int step=5, 
          const int block_size=5);

        /**
          * @brief Destructor
          */
        virtual ~VLDSIFT();

        /**
          * @brief Extract Dense SIFT features from a 2D blitz::Array, and save 
          *   the resulting features in the dst 2D blitz::Arrays.
          * @warning The src and dst arrays should have the correct size 
          *   (for dst the expected size is (getNKeypoints(), getDescriptorSize())
          *   An exception is thrown otherwise.
          */
        void operator()(const blitz::Array<float,2>& src, 
          blitz::Array<float,2>& dst);

        /**
          * @brief Returns the number of keypoints given the current parameters
          * when processing an image of the expected size.
          */
        inline size_t getNKeypoints() const 
        { return vl_dsift_get_keypoint_num(m_filt); }

        /**
          * @brief Returns the current size of a descriptor for a given keypoint 
          * given the current parameters.
          * (number of bins = n_blocks_along_X x n_blocks_along_Y x n_hist_bins
          */
        inline size_t getDescriptorSize() const 
        { return vl_dsift_get_descriptor_size(m_filt); }

    	protected:
        /**
          * @brief Attributes
          */
        int m_height;
        int m_width;
        int m_step;
        int m_block_size;
        VlDsiftFilter *m_filt;
    };

  }
}

#endif /* BOB_IP_VLDSIFT_H */
