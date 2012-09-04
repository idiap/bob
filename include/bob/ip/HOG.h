/**
 * @file cxx/ip/ip/HOG.h
 * @date Sun Apr 22 18:43:15 2012 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Computes Histogram of Oriented Gradients (HOG) descriptors
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

#ifndef BOB_IP_HOG_H
#define BOB_IP_HOG_H

#include "bob/core/array_assert.h"
#include "bob/ip/Exception.h"
#include "bob/ip/BlockCellGradientDescriptors.h"
#include <boost/shared_ptr.hpp>

namespace bob {
/**
 * \ingroup libip_api
 * @{
 */
  namespace ip {

    /**
      * @brief Function which computes an Histogram of Gradients for
      *   a given 'cell'. The inputs are the gradient magnitudes and the
      *   orientations for each pixel of the cell.
      *   The number of bins is given by the dimension of the output array. 
      * @param mag The input blitz array with the gradient magnitudes
      * @param ori The input blitz array with the orientations
      * @param hist The output blitz array which will contain the histogram 
      * @param init_hist Tells whether the output array should be 
      *   initialized to zero or not.
      * @param full_orientation Tells whether the full plane [0,360] is used
      *   or not (half plane [0,180] instead)
      * @warning Does not check that input arrays have same dimensions
      *   Assumes that the orientations are in range ([0,PI] or [0,2PI])
      */
    void hogComputeHistogram_(const blitz::Array<double,2>& mag, 
      const blitz::Array<double,2>& ori, blitz::Array<double,1>& hist, 
      const bool init_hist=true, const bool full_orientation=false);
    /**
      * @brief Function which computes an Histogram of Gradients for
      *   a given 'cell'. The inputs are the gradient magnitudes and the
      *   orientations for each pixel of the cell.
      *   The number of bins is given by the dimension of the output array. 
      * @param mag The input blitz array with the gradient magnitudes
      * @param ori The input blitz array with the orientations
      * @param hist The output blitz array which will contain the histogram 
      * @param init_hist Tells whether the output array should be 
      *   initialized to zero or not.
      * @param full_orientation Tell whether the full plane [0,360] is used
      *   or not (half plane [0,180] instead)
      */
    void hogComputeHistogram(const blitz::Array<double,2>& mag, 
      const blitz::Array<double,2>& ori, blitz::Array<double,1>& hist, 
      const bool init_hist=true, const bool full_orientation=false);

    /**
      * @brief Class to extract Histogram of Gradients (HOG) descriptors
      * This implementation relies on the following article,
      * "Histograms of Oriented Gradients for Human Detection",
      * N. Dalal, B. Triggs, in proceedings of the IEEE Conf. on Computer
      * Vision and Pattern Recognition, 2005.
      * Few remarks:
      *  1) Only single channel inputs (a.k.a. grayscale) are considered.
      *     Therefore, it does not take the maximum gradient over several 
      *     channels as proposed in the above article.
      *  2) Gamma/Colour normalization is not part of the descriptor 
      *     computation. However, this can easily be done (using this library)
      *     before extracting the descriptors.
      *  3) Gradients are computed using standard 1D centered gradient (except
      *     at the borders where the gradient is uncentered [-1 1]). This
      *     is the method which achieved best performance reported in the 
      *     article.
      *     To avoid too many uncentered gradients to be used, the gradients
      *     are computed on the full image prior to the cell decomposition.
      *     This implies that extra-pixels at each boundary of the cell are 
      *     contributing to the gradients, although these pixels are not 
      *     located inside the cell.
      *  4) R-HOG blocks (rectangular) normalisation is supported, but
      *     not C-HOG blocks (circular).
      *  5) Due to the similarity with the SIFT descriptors, this can also be
      *     used to extract dense-SIFT features.
      *  6) The first bin of each histogram is always centered around 0. This
      *     implies that the 'orientations are in [0-e,180-e]' rather than
      *     [0,180], e being half the angle size of a bin (same with [0,360]).
      */ 
    template <typename T>
    class HOG: public BlockCellGradientDescriptors<T,double>
    {
      public:
        /**
          * Constructor
          */
        HOG(const size_t height, const size_t width, 
          const size_t cell_dim=8, const bool full_orientation=false,
          const size_t cell_y=4, const size_t cell_x=4, 
          const size_t cell_ov_y=0, const size_t cell_ov_x=0,
          const size_t block_y=4, const size_t block_x=4, 
          const size_t block_ov_y=0, const size_t block_ov_x=0);

        /**
          * Copy constructor
          */
        HOG(const HOG& other);

        /**
          * Destructor
          */
        virtual ~HOG() {}

        /**
          * @brief Assignment operator
          */
        HOG& operator=(const HOG& other);

        /**
          * @brief Equal to
          */
        bool operator==(const HOG& b) const;
        /**
          * @brief Not equal to
          */
        bool operator!=(const HOG& b) const; 
 
        /**
          * Getters
          */
        bool getFullOrientation() const { return m_full_orientation; }
        /**
          * Setters
          */
        void setFullOrientation(const bool full_orientation)
        { m_full_orientation = full_orientation; }

        /**
          * Processes an input array. This extracts HOG descriptors from the
          * input image. The output is 3D, the first two dimensions being the 
          * y- and x- indices of the block, and the last one the index of the
          * bin (among the concatenated cell histograms for this block).
          */
        virtual void forward_(const blitz::Array<T,2>& input, 
          blitz::Array<double,3>& output);
        virtual void forward(const blitz::Array<T,2>& input, 
          blitz::Array<double,3>& output);

      protected:
        bool m_full_orientation;
    };

    template <typename T>
    HOG<T>::HOG(const size_t height, 
        const size_t width, const size_t cell_dim, 
        const bool full_orientation, 
        const size_t cell_y, const size_t cell_x, 
        const size_t cell_ov_y, const size_t cell_ov_x,
        const size_t block_y, const size_t block_x, 
        const size_t block_ov_y, const size_t block_ov_x):
      BlockCellGradientDescriptors<T,double>(height, width, 
        cell_dim, cell_y, cell_x, cell_ov_y, cell_ov_x, 
        block_y, block_x, block_ov_y, block_ov_x),
      m_full_orientation(full_orientation)
    {
    }

    template <typename T>
    HOG<T>::HOG(const HOG& other):
      BlockCellGradientDescriptors<T,double>(other),
      m_full_orientation(other.m_full_orientation)
    {      
    }

    template <typename T>
    HOG<T>& HOG<T>::operator=(const HOG<T>& other)
    {
      if(this != &other)
      {
        BlockCellGradientDescriptors<T,double>::operator=(other);
        m_full_orientation = other.m_full_orientation;
      }
      return *this;
    }

    template <typename T>
    bool HOG<T>::operator==(const HOG<T>& b) const
    {
      return (BlockCellGradientDescriptors<T,double>::operator==(b) && 
              this->m_full_orientation == b.m_full_orientation);
    }

    template <typename T>
    bool HOG<T>::operator!=(const HOG<T>& b) const
    {
      return !(this->operator==(b));
    }

    template <typename T>
    void HOG<T>::forward_(const blitz::Array<T,2>& input, 
      blitz::Array<double,3>& output)
    {
      BlockCellGradientDescriptors<T,double>::computeGradientMaps(input);
      // Computes the histograms for each cell
      BlockCellDescriptors<T,double>::m_cell_descriptor = 0.;
      blitz::Range rall = blitz::Range::all();
      for(size_t cy=0; cy<BlockCellDescriptors<T,double>::m_nb_cells_y; ++cy)
        for(size_t cx=0; cx<BlockCellDescriptors<T,double>::m_nb_cells_x; 
          ++cx)
        {
          blitz::Array<double,1> hist = 
            BlockCellDescriptors<T,double>::m_cell_descriptor(cy,cx,rall);
          blitz::Array<double,2> mag = 
            BlockCellGradientDescriptors<T,double>::m_cell_magnitude(cy,cx,
                                                              rall,rall);
          blitz::Array<double,2> ori = 
            BlockCellGradientDescriptors<T,double>::m_cell_orientation(cy,cx,
                                                              rall,rall);
          hogComputeHistogram_(mag, ori, hist, false, 
            m_full_orientation);
        }

      BlockCellDescriptors<T,double>::normalizeBlocks(output);
    }

    template <typename T>
    void HOG<T>::forward(const blitz::Array<T,2>& input, 
      blitz::Array<double,3>& output)
    {
      // Checks input/output arrays
      const blitz::TinyVector<int,3> r = 
        BlockCellDescriptors<T,double>::getOutputShape();
      bob::core::array::assertSameShape(output, r);

      // Generates the HOG descriptors
      forward_(input, output);
    }

  }

/**
 * @}
 */
}

#endif /* BOB_IP_HOG_H */
