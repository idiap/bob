/**
 * @file bob/ip/BlockCellGradientDescriptors.h
 * @date Sun Apr 22 16:03:15 2012 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Abstract class for extracting gradient-based descriptors by 
 *   decomposing an image (or an image patch) into a set of cells, and blocks.
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

#ifndef BOB_IP_BLOCK_CELL_GRADIENT_DESCRIPTORS_H
#define BOB_IP_BLOCK_CELL_GRADIENT_DESCRIPTORS_H

#include "bob/core/array_assert.h"
#include "bob/ip/Exception.h"
#include "bob/ip/BlockCellDescriptors.h"
#include "bob/ip/block.h"
#include "bob/math/gradient.h"
#include <boost/shared_ptr.hpp>

namespace bob {
  /**
    * \ingroup libip_api
    * @{
    */
  namespace ip {

    /**
      * Gradient 'magnitude' used
      * - Magnitude: L2 magnitude over X and Y
      * - MagnitudeSquare: Square of the L2 magnitude
      * - SqrtMagnitude: Square root of the L2 magnitude
      */
    typedef enum GradientMagnitudeType_ 
    { Magnitude, MagnitudeSquare, SqrtMagnitude } GradientMagnitudeType;

    /**
      * @brief Class to extract gradient magnitude and orientation maps
      */
    class GradientMaps
    {
      public:
        /**
          * Constructor
          */
        GradientMaps(const size_t height, const size_t width, 
          const GradientMagnitudeType mag_type=Magnitude);
        /**
          * Copy constructor
          */
        GradientMaps(const GradientMaps& other);
        /**
          * Destructor
          */
        virtual ~GradientMaps() {}

        /**
         * @brief Assignment operator
         */
        GradientMaps& operator=(const GradientMaps& other);
        /**
         * @brief Equal to
         */
        bool operator==(const GradientMaps& b) const;
        /**
         * @brief Not equal to
         */
        bool operator!=(const GradientMaps& b) const; 
 
        /**
          * Sets the height
          */
        void setHeight(const size_t height);
        /**
          * Sets the width 
          */
        void setWidth(const size_t width);
        /**
          * Resizes the cache
          */
        void resize(const size_t height, const size_t width);
        /**
          * Sets the magnitude type to use
          */
        void setGradientMagnitudeType(const GradientMagnitudeType mag_type)
        { m_mag_type = mag_type; }
        /**
          * Returns the current height
          */
        size_t getHeight() const { return m_gy.extent(0); }
        /**
          * Returns the current width
          */
        size_t getWidth() const { return m_gy.extent(1); }
        /**
          * Returns the magnitude type used
          */
        GradientMagnitudeType getGradientMagnitudeType() const
        { return m_mag_type; }

        /**
          * Processes an input array
          */
        template <typename T>
        void forward(const blitz::Array<T,2>& input, 
          blitz::Array<double,2>& magnitude, 
          blitz::Array<double,2>& orientation);
        template <typename T>
        void forward_(const blitz::Array<T,2>& input, 
          blitz::Array<double,2>& magnitude, 
          blitz::Array<double,2>& orientation);

      private:
        blitz::Array<double,2> m_gy;
        blitz::Array<double,2> m_gx;
        GradientMagnitudeType m_mag_type;
    };


    template <typename T>
    void GradientMaps::forward_(const blitz::Array<T,2>& input,
      blitz::Array<double,2>& magnitude, blitz::Array<double,2>& orientation)
    {
      // Computes the gradient
      bob::math::gradient<T,double>(input, m_gy, m_gx);

      // Computes the magnitude map
      switch(m_mag_type)
      {
        case MagnitudeSquare:
          magnitude = blitz::pow2(m_gy) + blitz::pow2(m_gx);
          break;
        case SqrtMagnitude:
          magnitude = blitz::sqrt(blitz::sqrt(blitz::pow2(m_gy) + 
                                  blitz::pow2(m_gx)));
          break;
        case Magnitude: 
        default: 
          magnitude = blitz::sqrt(blitz::pow2(m_gy) + blitz::pow2(m_gx));
      }
      // Computes the orientation map (range: [-PI,PI])
      orientation = blitz::atan2(m_gy, m_gx);
    }
 
    template <typename T>
    void GradientMaps::forward(const blitz::Array<T,2>& input,
      blitz::Array<double,2>& magnitude, blitz::Array<double,2>& orientation)
    {
      // Checks input/output arrays
      bob::core::array::assertSameShape(input, m_gy);
      bob::core::array::assertSameShape(magnitude, m_gy);
      bob::core::array::assertSameShape(orientation, m_gy);

      // Calls the HOGGradientMaps extractor
      forward_(input, magnitude, orientation);
    }


    /**
      * @brief Abstract class to extract Gradient-based descriptors using a 
      *   decomposition into cells (unormalized descriptors) and blocks 
      *   (groups of cells used for normalization purpose)
      */
    template <typename T, typename U>
    class BlockCellGradientDescriptors: public BlockCellDescriptors<T,U>
    {
      public:
        /**
          * Constructor
          */
        BlockCellGradientDescriptors(const size_t height, const size_t width, 
          const size_t cell_dim=8, 
          const size_t cell_y=4, const size_t cell_x=4, 
          const size_t cell_ov_y=0, const size_t cell_ov_x=0,
          const size_t block_y=4, const size_t block_x=4, 
          const size_t block_ov_y=0, const size_t block_ov_x=0);

        /**
          * Copy constructor
          */
        BlockCellGradientDescriptors(const BlockCellGradientDescriptors& b);

        /**
          * Destructor
          */
        virtual ~BlockCellGradientDescriptors() {}

        /**
         * @brief Assignment operator
         */
        BlockCellGradientDescriptors& operator=(
          const BlockCellGradientDescriptors& other);
        /**
         * @brief Equal to
         */
        bool operator==(const BlockCellGradientDescriptors& b) const;
        /**
         * @brief Not equal to
         */
        bool operator!=(const BlockCellGradientDescriptors& b) const; 
 
        /**
          * Getters
          */
        GradientMagnitudeType getGradientMagnitudeType() const
        { return m_gradient_maps->getGradientMagnitudeType(); }
        /**
          * Setters
          */
        void setGradientMagnitudeType(const GradientMagnitudeType m)
        { m_gradient_maps->setGradientMagnitudeType(m); }

        /**
          * Processes an input array. This extracts HOG descriptors from the
          * input image. The output is 3D, the first two dimensions being the 
          * y- and x- indices of the block, and the last one the index of the
          * bin (among the concatenated cell histograms for this block).
          */
        virtual void forward_(const blitz::Array<T,2>& input, 
          blitz::Array<U,3>& output) = 0;
        virtual void forward(const blitz::Array<T,2>& input, 
          blitz::Array<U,3>& output) = 0;

      protected:
        /**
          * Computes the gradient maps, and their decomposition into cells
          */
        virtual void computeGradientMaps(const blitz::Array<T,2>& input);

        // Methods to resize arrays in cache
        virtual void resizeCache();
        virtual void resizeCellCache();
        
        // Gradient related
        boost::shared_ptr<GradientMaps> m_gradient_maps;
        // Gradient maps for magnitude and orientation
        blitz::Array<double,2> m_magnitude;
        blitz::Array<double,2> m_orientation;
        // Gradient maps decomposed into blocks
        blitz::Array<double,4> m_cell_magnitude;
        blitz::Array<double,4> m_cell_orientation; 
    };

    template <typename T, typename U>
    BlockCellGradientDescriptors<T,U>::BlockCellGradientDescriptors(
        const size_t height, const size_t width, const size_t cell_dim, 
        const size_t cell_y, const size_t cell_x, 
        const size_t cell_ov_y, const size_t cell_ov_x,
        const size_t block_y, const size_t block_x, 
        const size_t block_ov_y, const size_t block_ov_x):
      BlockCellDescriptors<T,U>(height, width, cell_dim, cell_y, cell_x,
        cell_ov_y, cell_ov_x, block_y, block_x, block_ov_y, block_ov_x),
      m_gradient_maps(new GradientMaps(height, width))
    {
      resizeCache();
    }

    template <typename T, typename U>
    BlockCellGradientDescriptors<T,U>::BlockCellGradientDescriptors(
        const BlockCellGradientDescriptors<T,U>& b):
      BlockCellDescriptors<T,U>(b), 
      m_gradient_maps(new GradientMaps(b.m_height, b.m_width, 
                            b.getGradientMagnitudeType()))
    {
      resizeCache();
    }

    template <typename T, typename U>
    BlockCellGradientDescriptors<T,U>&
    BlockCellGradientDescriptors<T,U>::operator=(
        const BlockCellGradientDescriptors<T,U>& other)
    {
      if(this != &other)
      {
        BlockCellDescriptors<T,U>::operator=(other);
        m_gradient_maps.reset(new GradientMaps(other.m_height, other.m_width,
                                          other.getGradientMagnitudeType()));
        resizeCache();
      }
      return *this;
    }

    template <typename T, typename U>
    bool BlockCellGradientDescriptors<T,U>::operator==(
        const BlockCellGradientDescriptors<T,U>& b) const
    {
      return (BlockCellDescriptors<T,U>::operator==(b) && 
              *(this->m_gradient_maps) == *(b.m_gradient_maps));
    }
 
    template <typename T, typename U>
    bool BlockCellGradientDescriptors<T,U>::operator!=(
        const BlockCellGradientDescriptors<T,U>& b) const
    {
      return !(this->operator==(b));
    }
 
    template <typename T, typename U>
    void BlockCellGradientDescriptors<T,U>::resizeCache()
    {
      // Resizes BlockCellDescriptors first
      BlockCellDescriptors<T,U>::resizeCache();
      // Resizes everything else
      m_gradient_maps->resize(BlockCellDescriptors<T,U>::m_height, 
        BlockCellDescriptors<T,U>::m_width);
      m_magnitude.resize(BlockCellDescriptors<T,U>::m_height, 
        BlockCellDescriptors<T,U>::m_width);
      m_orientation.resize(BlockCellDescriptors<T,U>::m_height, 
        BlockCellDescriptors<T,U>::m_width);
    }

    template <typename T, typename U>
    void BlockCellGradientDescriptors<T,U>::resizeCellCache()
    {
      // Resizes BlockCellDescriptors first
      BlockCellDescriptors<T,U>::resizeCellCache();
      // Resizes everything else
      m_cell_magnitude.resize(BlockCellDescriptors<T,U>::m_nb_cells_y, 
        BlockCellDescriptors<T,U>::m_nb_cells_x, 
        BlockCellDescriptors<T,U>::m_cell_y, 
        BlockCellDescriptors<T,U>::m_cell_x);
      m_cell_orientation.resize(BlockCellDescriptors<T,U>::m_nb_cells_y, 
        BlockCellDescriptors<T,U>::m_nb_cells_x, 
        BlockCellDescriptors<T,U>::m_cell_y, 
        BlockCellDescriptors<T,U>::m_cell_x);
    }

    template <typename T, typename U>
    void BlockCellGradientDescriptors<T,U>::computeGradientMaps(
      const blitz::Array<T,2>& input)
    {
      // Computes the Gradients maps (magnitude and orientation)
      m_gradient_maps->forward_(input, m_magnitude, m_orientation);
      
      // Performs the block decomposition on the Gradients maps
      block(m_magnitude, m_cell_magnitude, 
        BlockCellDescriptors<T,U>::m_cell_y, 
        BlockCellDescriptors<T,U>::m_cell_x,
        BlockCellDescriptors<T,U>::m_cell_ov_y, 
        BlockCellDescriptors<T,U>::m_cell_ov_x);
      block(m_orientation, m_cell_orientation, 
        BlockCellDescriptors<T,U>::m_cell_y, 
        BlockCellDescriptors<T,U>::m_cell_x,
        BlockCellDescriptors<T,U>::m_cell_ov_y, 
        BlockCellDescriptors<T,U>::m_cell_ov_x); 
    }

  }

/**
 * @}
 */
}

#endif /* BOB_IP_BLOCK_CELL_GRADIENT_DESCRIPTORS_H */
