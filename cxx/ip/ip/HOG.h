/**
 * @file cxx/ip/ip/HOG.h
 * @date Sat Apr 14 21:01:15 2011 +0200
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

#include "core/array_assert.h"
#include "ip/Exception.h"
#include "ip/block.h"
#include "math/gradient.h"
#include <boost/shared_ptr.hpp>

namespace bob {
/**
 * \ingroup libip_api
 * @{
 */
  namespace ip {

    namespace hog {
      /**
        * Gradient 'magnitude' used
        * - Magnitude: L2 magnitude over X and Y
        * - MagnitudeSquare: Square of the L2 magnitude
        * - SqrtMagnitude: Square root of the L2 magnitude
        */
      typedef enum GradientMagnitudeType_ 
      { Magnitude, MagnitudeSquare, SqrtMagnitude } GradientMagnitudeType;
      
      /**
        * Norm used for normalizing the HOG blocks
        * - L2: Euclidean norm
        * - L2Hys: L2 norm with clipping of high values
        * - L1: L1 norm (Manhattan distance)
        * - L1sqrt: Square root of the L1 norm
        * - None: no norm used
        */
      typedef enum BlockNorm_ { L2, L2Hys, L1, L1sqrt, None } BlockNorm;
    }

    namespace detail {
      /**
        * Vectorizes an histogram and multiply values by a constant factor
        */
      template <typename T>
      void hogVectorizeMultHist(const blitz::Array<T,3> in, 
        blitz::Array<T,1> out, const T factor=1)
      {
        int n_cells_y = in.extent(0);
        int n_cells_x = in.extent(1);
        int n_bins = in.extent(2);
        blitz::Range rall = blitz::Range::all();
        for(int cy=0; cy<n_cells_y; ++cy)
          for(int cx=0; cx<n_cells_x; ++cx)
          {
            blitz::Array<T,1> in_ = in(cy,cx,rall);
            blitz::Array<T,1> out_ = out(blitz::Range(
                  (cy*n_cells_x+cx)*n_bins,(cy*n_cells_x+cx+1)*n_bins-1));
            out_ = in_ * factor;
          }
      }

      template <typename T>
      void hogVectorizeMultHist(const blitz::Array<T,2> in, 
        blitz::Array<T,1> out, const T factor=1)
      {
        int n_cells = in.extent(0);
        int n_bins = in.extent(1);
        blitz::Range rall = blitz::Range::all();
        for(int c=0; c<n_cells; ++c)
        {
          blitz::Array<T,1> in_ = in(c,rall);
          blitz::Array<T,1> out_ = out(blitz::Range(
                c*n_bins,(c+1)*n_bins-1));
          out_ = in_ * factor;
        }
      }

      template <typename T>
      void hogVectorizeMultHist(const blitz::Array<T,1> in, 
        blitz::Array<T,1> out, const T factor=1)
      {
        out = in * factor;
      }
    }

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
    void hogComputeCellHistogram_(const blitz::Array<double,2>& mag, 
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
    void hogComputeCellHistogram(const blitz::Array<double,2>& mag, 
      const blitz::Array<double,2>& ori, blitz::Array<double,1>& hist, 
      const bool init_hist=true, const bool full_orientation=false);
    
    /**
      * @brief Function which normalizes a set of cells (Histogram of 
      *   Gradients), and returns the corresponding block descriptor.
      *   The number of bins is given by the dimension of the output array. 
      * @param hist The input cells (last dimension is for the histogram bins.
      * @param norm_hist The output 1D normalized block descriptor
      * @param block_norm The norm used by the procedure 
      * @param eps The epsilon used for the block normalization 
      *   (to avoid zero norm, as in the article of Dalal and Triggs) 
      * @param threshold The threshold used for the block normalization
      *   This is only used with the L2Hys norm, for the clipping of large 
      *   values.
      * @warning Does not check that input and output arrays have the same
      *   number of elements.
      */ 
    template <int D>
    void hogNormalizeBlock_(const blitz::Array<double,D>& hist,
      blitz::Array<double,1>& norm_hist, const hog::BlockNorm block_norm=hog::L2,
      const double eps=1e-10, const double threshold=0.2)
    {
      // Use multiplication rather than inversion (should be faster)
      double sumInv;
      switch(block_norm)
      {
        case hog::None:
          bob::ip::detail::hogVectorizeMultHist(hist, norm_hist);
          break;
        case hog::L2Hys:
          // Normalizes to unit length (using L2)
          sumInv = 1. / sqrt(blitz::sum(blitz::pow2(hist)) + eps*eps);
          bob::ip::detail::hogVectorizeMultHist(hist, norm_hist, sumInv);
          // Clips values above threshold
          norm_hist = blitz::where(norm_hist <= threshold, norm_hist, threshold);
          // Normalizes to unit length (using L2)
          sumInv = 1. / sqrt(blitz::sum(blitz::pow2(norm_hist)) + eps*eps);
          norm_hist = norm_hist * sumInv;
          break;
        case hog::L1:
          // Normalizes to unit length (using L1)
          sumInv = 1. / (blitz::sum(blitz::abs(hist)) + eps);
          bob::ip::detail::hogVectorizeMultHist(hist, norm_hist, sumInv);
          break;
        case hog::L1sqrt:
          // Normalizes to unit length (using L1)
          sumInv = 1. / (blitz::sum(blitz::abs(hist)) + eps);
          bob::ip::detail::hogVectorizeMultHist(hist, norm_hist, sumInv);
          norm_hist = blitz::sqrt(norm_hist);
          break;
        case hog::L2:
        default:
          // Normalizes to unit length (using L2)
          sumInv = 1. / sqrt(blitz::sum(blitz::pow2(hist)) + eps*eps);
          bob::ip::detail::hogVectorizeMultHist(hist, norm_hist, sumInv);
          break;
      }
    }
    /**
      * @brief Function which normalizes a set of cells (Histogram of 
      *   Gradients), and returns the corresponding block descriptor.
      *   The number of bins is given by the dimension of the output array. 
      * @param hist The input cells (last dimension is for the histogram bins.
      * @param norm_hist The output 1D normalized block descriptor
      * @param block_norm The norm used by the procedure 
      * @param eps The epsilon used for the block normalization 
      *   (to avoid zero norm, as in the article of Dalal and Triggs) 
      * @param threshold The threshold used for the block normalization
      *   This is only used with the L2Hys norm, for the clipping of large 
      *   values.
      */ 
    template <int D>
    void hogNormalizeBlock(const blitz::Array<double,D>& hist,
      blitz::Array<double,1>& norm_hist, const hog::BlockNorm block_norm=hog::L2,
      const double eps=1e-10, const double threshold=0.2)
    {
      // Checks input/output arrays
      int nhist=1;
      for(int d=0; d<D; ++d) nhist *= hist.extent(d);
      bob::core::array::assertSameDimensionLength(nhist, norm_hist.extent(0));

      // Normalizes
      hogNormalizeBlock_(hist, norm_hist, block_norm, eps, threshold);
    }

    /**
      * @brief Class to extract gradient magnitude and orientation maps
      */
    class HOGGradientMaps
    {
      public:
        /**
          * Constructor
          */
        HOGGradientMaps(const size_t height, const size_t width, 
          const hog::GradientMagnitudeType mag_type=hog::Magnitude);
        /**
          * Destructor
          */
        ~HOGGradientMaps() {}
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
        void setGradientMagnitudeType(const hog::GradientMagnitudeType mag_type);
        /**
          * Returns the current height
          */
        inline const size_t getHeight() const { return m_gy.extent(0); }
        /**
          * Returns the current width
          */
        inline const size_t getWidth() const { return m_gy.extent(1); }
        /**
          * Returns the magnitude type used
          */
        inline const hog::GradientMagnitudeType 
        getGradientMagnitudeType() const
        { return m_mag_type; }
        /**
          * Processes an input array
          */
        template <typename T>
        void forward(const blitz::Array<T,2>& input, 
          blitz::Array<double,2>& magnitude, blitz::Array<double,2>& orientation);
        template <typename T>
        void forward_(const blitz::Array<T,2>& input, 
          blitz::Array<double,2>& magnitude, blitz::Array<double,2>& orientation);

      private:
        blitz::Array<double,2> m_gy;
        blitz::Array<double,2> m_gx;
        hog::GradientMagnitudeType m_mag_type;
    };


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
    class HOG
    {
      public:
        /**
          * Constructor
          */
        HOG(const size_t height, const size_t width, 
          const size_t nb_bins=8, const bool full_orientation=false, 
          const size_t cell_y=4, const size_t cell_x=4, 
          const size_t cell_ov_y=0, const size_t cell_ov_x=0,
          const size_t block_y=4, const size_t block_x=4, 
          const size_t block_ov_y=0, const size_t block_ov_x=0);
        /**
          * Destructor
          */
        ~HOG() {}

        /**
          * Resizes the cache
          */
        void resize(const size_t height, const size_t width);
 
        /**
          * Getters
          */
        inline size_t getHeight() const { return m_height; }
        inline size_t getWidth() const { return m_width; }
        inline hog::GradientMagnitudeType getGradientMagnitudeType() const
        { return m_hog_gradient_maps->getGradientMagnitudeType(); }
        inline size_t getNBins() const { return m_nb_bins; }
        inline bool getFullOrientation() const { return m_full_orientation; }
        inline size_t getCellHeight() const { return m_cell_y; }
        inline size_t getCellWidth() const { return m_cell_x; }
        inline size_t getCellOverlapHeight() const { return m_cell_ov_y; }
        inline size_t getCellOverlapWidth() const { return m_cell_ov_x; }
        inline size_t getBlockHeight() const { return m_block_y; }
        inline size_t getBlockWidth() const { return m_block_x; }
        inline size_t getBlockOverlapHeight() const { return m_block_ov_y; }
        inline size_t getBlockOverlapWidth() const { return m_block_ov_x; }
        inline hog::BlockNorm getBlockNorm() const { return m_block_norm; }
        inline double getBlockNormEps() const { return m_block_norm_eps; }
        inline double getBlockNormThreshold() const { return m_block_norm_threshold; }
        /**
          * Setters
          */
        void setHeight(const size_t height)
        { m_height = height; resizeCache(); }
        void setWidth(const size_t width)
        { m_width = width; resizeCache(); }
        void setGradientMagnitudeType(const hog::GradientMagnitudeType m)
        { m_hog_gradient_maps->setGradientMagnitudeType(m); }
        void setNBins(const size_t nb_bins)
        { m_nb_bins = nb_bins; resizeCellCache(); }
        void setFullOrientation(const bool full_orientation)
        { m_full_orientation = full_orientation; }
        void setCellHeight(const size_t cell_y)
        { m_cell_y = cell_y; resizeCellCache(); }
        void setCellWidth(const size_t cell_x)
        { m_cell_x = cell_x; resizeCellCache(); }
        void setCellOverlapHeight(const size_t cell_ov_y)
        { m_cell_ov_y = cell_ov_y; resizeCellCache(); }
        void setCellOverlapWidth(const size_t cell_ov_x)
        { m_cell_ov_x = cell_ov_x; resizeCellCache(); }
        void setBlockHeight(const size_t block_y)
        { m_block_y = block_y; resizeBlockCache(); }
        void setBlockWidth(const size_t block_x)
        { m_block_x = block_x; resizeBlockCache(); }
        void setBlockOverlapHeight(const size_t block_ov_y)
        { m_block_ov_y = block_ov_y; resizeBlockCache(); }
        void setBlockOverlapWidth(const size_t block_ov_x)
        { m_block_ov_x = block_ov_x; resizeBlockCache(); }
        void setBlockNorm(const hog::BlockNorm block_norm)
        { m_block_norm = block_norm; }
        void setBlockNormEps(const double block_norm_eps)
        { m_block_norm_eps = block_norm_eps; }
        void setBlockNormThreshold(const double block_norm_threshold)
        { m_block_norm_threshold = block_norm_threshold; }

        /**
          * Disable block normalization. This is performed by setting parameters
          * such that the cells are not further processed, that is
          * block_y=1, block_x=1, block_ov_y=0, block_ov_x=0, and 
          * block_norm=hog::None.
          */
        void disableBlockNormalization();

        /**
          * Gets the descriptor output size given the current parameters and
          * size. (number of blocks along Y x number of block along X x number of bins)
          */
        const blitz::TinyVector<int,3> getOutputShape() const;
        /**
          * Processes an input array. This extracts HOG descriptors from the
          * input image. The output is 3D, the first two dimensions being the 
          * y- and x- indices of the block, and the last one the index of the
          * bin (among the concatenated cell histograms for this block).
          */
        template <typename T>
        void forward_(const blitz::Array<T,2>& input, 
          blitz::Array<double,3>& output);
        template <typename T>
        void forward(const blitz::Array<T,2>& input, 
          blitz::Array<double,3>& output);

      private:
        // Methods to resize arrays in cache
        void resizeCache();
        void resizeCellCache();
        void resizeBlockCache();

        // Input size
        size_t m_height;
        size_t m_width;
        // Gradient related
        boost::shared_ptr<ip::HOGGradientMaps> m_hog_gradient_maps;
        // Histogram-related variables
        size_t m_nb_bins;
        bool m_full_orientation;
        // Cell-related variables
        size_t m_cell_y;
        size_t m_cell_x;
        size_t m_cell_ov_y;
        size_t m_cell_ov_x;
        // Block-related variables (normalization)
        bool m_block_normalization;
        size_t m_block_y;
        size_t m_block_x;
        size_t m_block_ov_y;
        size_t m_block_ov_x;
        hog::BlockNorm m_block_norm;
        double m_block_norm_eps;
        double m_block_norm_threshold;

        // Cache
        // Number of blocks along Y- and X- axes
        size_t m_nb_blocks_y;
        size_t m_nb_blocks_x;

        // Gradient maps for magnitude and orientation
        blitz::Array<double,2> m_magnitude;
        blitz::Array<double,2> m_orientation;
        // Gradient maps decomposed into blocks
        blitz::Array<double,4> m_cell_magnitude;
        blitz::Array<double,4> m_cell_orientation;
        // Non-normalized HOG computed at the cell level
        blitz::Array<double,3> m_cell_hist;
    };

    template <typename T>
    void HOGGradientMaps::forward_(const blitz::Array<T,2>& input,
      blitz::Array<double,2>& magnitude, blitz::Array<double,2>& orientation)
    {
      // Computes the gradient
      bob::math::gradient<T,double>(input, m_gy, m_gx);

      // Computes the magnitude map
      switch(m_mag_type)
      {
        case hog::MagnitudeSquare:
          magnitude = blitz::pow2(m_gy) + blitz::pow2(m_gx);
          break;
        case hog::SqrtMagnitude:
          magnitude = blitz::sqrt(blitz::sqrt(blitz::pow2(m_gy) + blitz::pow2(m_gx)));
          break;
        case hog::Magnitude: 
        default: 
          magnitude = blitz::sqrt(blitz::pow2(m_gy) + blitz::pow2(m_gx));
      }
      // Computes the orientation map (range: [-PI,PI])
      orientation = blitz::atan2(m_gx, m_gy);
    }
 
    template <typename T>
    void HOGGradientMaps::forward(const blitz::Array<T,2>& input,
      blitz::Array<double,2>& magnitude, blitz::Array<double,2>& orientation)
    {
      // Checks input/output arrays
      bob::core::array::assertSameShape(input, m_gy);
      bob::core::array::assertSameShape(magnitude, m_gy);
      bob::core::array::assertSameShape(orientation, m_gy);

      // Calls the HOGGradientMaps extractor
      forward_(input, magnitude, orientation);
    }

    template <typename T>
    void HOG::forward_(const blitz::Array<T,2>& input, 
      blitz::Array<double,3>& output)
    {
      // Computes the Gradients maps (magnitude and orientation)
      m_hog_gradient_maps->forward_(input, m_magnitude, m_orientation);
      
      // Extracts the cells and compute the descriptors
      // a/ Performs the block decomposition on the Gradients maps
      bob::ip::block(m_magnitude, m_cell_magnitude, m_cell_y, m_cell_x,
        m_cell_ov_y, m_cell_ov_x);
      bob::ip::block(m_orientation, m_cell_orientation, m_cell_y, m_cell_x,
        m_cell_ov_y, m_cell_ov_x);
      // b/ Computes the histograms for each cell
      m_cell_hist = 0.;
      blitz::Range rall = blitz::Range::all();
      for(size_t cy=0; (int)cy<m_cell_hist.extent(0); ++cy)
        for(size_t cx=0; (int)cx<m_cell_hist.extent(1); ++cx)
        {
          blitz::Array<double,1> hist = m_cell_hist(cy,cx,rall);
          blitz::Array<double,2> mag = m_cell_magnitude(cy,cx,rall,rall);
          blitz::Array<double,2> ori = m_cell_orientation(cy,cx,rall,rall);
          bob::ip::hogComputeCellHistogram_(mag, ori, hist, false, 
            m_full_orientation);
        }

      // Normalizes by block
      for(size_t by=0; by<m_nb_blocks_y; ++by)
        for(size_t bx=0; bx<m_nb_blocks_x; ++bx)
        {
          blitz::Range ry(by,by+m_block_y-1);
          blitz::Range rx(bx,bx+m_block_x-1);
          blitz::Array<double,3> cells_block = m_cell_hist(ry,rx,rall);
          blitz::Array<double,1> block = output(by,bx,rall);
          bob::ip::hogNormalizeBlock_(cells_block, block, m_block_norm, 
            m_block_norm_eps, m_block_norm_threshold);
        }
    }

    template <typename T>
    void HOG::forward(const blitz::Array<T,2>& input, 
      blitz::Array<double,3>& output)
    {
      // Checks input/output arrays
      const blitz::TinyVector<int,3> r = getOutputShape();
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
