/**
 * @file cxx/ip/ip/VLDSIFT.h
 * @date Mon Jan 23 20:46:07 2012 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file defines classes to compute SIFT features using VLFeat
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
        VLDSIFT(const size_t height, const size_t width, const size_t step=5, 
          const size_t block_size=5);

        /**
          * @brief Copy constructor
          */
        VLDSIFT(const VLDSIFT& other);

        /**
          * @brief Destructor
          */
        virtual ~VLDSIFT();

        /**
          * @brief Assignment operator
          */
        VLDSIFT& operator=(const VLDSIFT& other);

        /**
          * @brief Equal to
          */
        bool operator==(const VLDSIFT& b) const;
        /**
          * @brief Not equal to
          */
        bool operator!=(const VLDSIFT& b) const; 

        /**
          * @brief Getters
          */
        size_t getHeight() const { return m_height; }
        size_t getWidth() const { return m_width; }
        size_t getStepY() const { return m_step_y; }
        size_t getStepX() const { return m_step_x; }
        size_t getBlockSizeY() const { return m_block_size_y; }
        size_t getBlockSizeX() const { return m_block_size_x; }
        bool getUseFlatWindow() const { return m_use_flat_window; }
        double getWindowSize() const { return m_window_size; }

        /**
          * @brief Setters
          */
        void setHeight(const size_t height) 
        { m_height = height; cleanup(); allocateAndSet(); }
        void setWidth(const size_t width) 
        { m_width = width; cleanup(); allocateAndSet(); }
        void setStepY(const size_t step_y) 
        { m_step_y = step_y; vl_dsift_set_steps(m_filt, m_step_x, m_step_y); }
        void setStepX(const size_t step_x) 
        { m_step_x = step_x; vl_dsift_set_steps(m_filt, m_step_x, m_step_y); }
        void setBlockSizeY(const size_t block_size_y);
        void setBlockSizeX(const size_t block_size_x);
        void setUseFlatWindow(const bool use) 
        { m_use_flat_window = use; vl_dsift_set_flat_window(m_filt, use); }
        void setWindowSize(const double size) 
        { m_window_size = size; vl_dsift_set_window_size(m_filt, size); }
 
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
        size_t getNKeypoints() const 
        { return vl_dsift_get_keypoint_num(m_filt); }

        /**
          * @brief Returns the current size of a descriptor for a given keypoint 
          * given the current parameters.
          * (number of bins = n_blocks_along_X x n_blocks_along_Y x n_hist_bins
          */
        size_t getDescriptorSize() const 
        { return vl_dsift_get_descriptor_size(m_filt); }

      protected:
        /**
          * @brief Allocation methods
          */
        void allocate();
        /**
          * @brief Resets the properties of the VLfeat filter object
          */
        void setFilterProperties(); 
        /**
          * @brief Reallocate and resets the properties of the VLfeat objects
          */
        void allocateAndSet();

        /**
          * @brief Deallocation method
          */
        void cleanup();

        /**
          * @brief Attributes
          */
        size_t m_height;
        size_t m_width;
        size_t m_step_y;
        size_t m_step_x;
        size_t m_block_size_y;
        size_t m_block_size_x;
        bool m_use_flat_window;
        double m_window_size;
        VlDsiftFilter *m_filt;
    };

  }
}

#endif /* BOB_IP_VLDSIFT_H */
