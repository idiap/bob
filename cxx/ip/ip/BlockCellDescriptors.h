/**
 * @file cxx/ip/ip/BlockCellDescriptors.h
 * @date Sun Apr 22 16:03:15 2012 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Abstract class for extracting descriptors by decomposing an image
 *   (or an image patch) into a set of cells, and blocks.
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

#ifndef BOB_IP_CELL_BLOCK_DESCRIPTORS_H
#define BOB_IP_CELL_BLOCK_DESCRIPTORS_H

#include "core/array_assert.h"
#include "ip/Exception.h"
#include "ip/block.h"
#include <boost/shared_ptr.hpp>

#include <iostream>

namespace bob {
/**
 * \ingroup libip_api
 * @{
 */
  namespace ip {

    namespace detail {
      /**
        * Vectorizes an array and multiply values by a constant factor
        */
      template <typename T>
      void vectorizeMultArray(const blitz::Array<T,3> in, 
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
      void vectorizeMultArray(const blitz::Array<T,2> in, 
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
      void vectorizeMultArray(const blitz::Array<T,1> in, 
        blitz::Array<T,1> out, const T factor=1)
      {
        out = in * factor;
      }
    }


    /**
      * Norm used for normalizing the descriptor blocks
      * - L2: Euclidean norm
      * - L2Hys: L2 norm with clipping of high values
      * - L1: L1 norm (Manhattan distance)
      * - L1sqrt: Square root of the L1 norm
      * - None: no norm used
      * TODO: ZeroMean/UnitVariance normalization?
      */
    typedef enum BlockNorm_ { L2, L2Hys, L1, L1sqrt, None } BlockNorm;


    /**
      * @brief Function which normalizes a set of cells, and returns the 
      *   corresponding 1D block descriptor.
      * @param descr The input descriptor (first two dimensions are for the 
      *   spatial location of the cell, whereas the length of the last 
      *   dimension corresponds to the dimensionality of the cell descriptor).
      * @param norm_descr The output 1D normalized block descriptor
      * @param block_norm The norm used by the procedure 
      * @param eps The epsilon used for the block normalization 
      *   (to avoid division by zero norm) 
      * @param threshold The threshold used for the block normalization
      *   This is only used with the L2Hys norm, for the clipping of large 
      *   values.
      * @warning Does not check that input and output arrays have the same
      *   number of elements.
      */ 
    template <typename U, int D>
    void normalizeBlock_(const blitz::Array<U,D>& descr,
      blitz::Array<U,1>& norm_descr, const BlockNorm block_norm=L2,
      const double eps=1e-10, const double threshold=0.2)
    {
      // Use multiplication rather than inversion (should be faster)
      double sumInv;
      switch(block_norm)
      {
        case None:
          bob::ip::detail::vectorizeMultArray(descr, norm_descr);
          break;
        case L2Hys:
          // Normalizes to unit length (using L2)
          sumInv = 1. / sqrt(blitz::sum(blitz::pow2(blitz::abs(descr))) + eps*eps);
          bob::ip::detail::vectorizeMultArray(descr, norm_descr, sumInv);
          // Clips values above threshold
          norm_descr = blitz::where(blitz::abs(norm_descr) <= threshold, norm_descr, threshold);
          // Normalizes to unit length (using L2)
          sumInv = 1. / sqrt(blitz::sum(blitz::pow2(blitz::abs(norm_descr))) + eps*eps);
          norm_descr = norm_descr * sumInv;
          break;
        case L1:
          // Normalizes to unit length (using L1)
          sumInv = 1. / (blitz::sum(blitz::abs(descr)) + eps);
          bob::ip::detail::vectorizeMultArray(descr, norm_descr, sumInv);
          break;
        case L1sqrt:
          // Normalizes to unit length (using L1)
          sumInv = 1. / (blitz::sum(blitz::abs(descr)) + eps);
          bob::ip::detail::vectorizeMultArray(descr, norm_descr, sumInv);
          norm_descr = blitz::sqrt(norm_descr);
          break;
        case L2:
        default:
          // Normalizes to unit length (using L2)
          sumInv = 1. / sqrt(blitz::sum(blitz::pow2(blitz::abs(descr))) + eps*eps);
          bob::ip::detail::vectorizeMultArray(descr, norm_descr, sumInv);
          break;
      }
    }

    /**
      * @brief Function which normalizes a set of cells, and returns the 
      *   corresponding 1D block descriptor.
      * @param descr The input descriptor (first two dimensions are for the 
      *   spatial location of the cell, whereas the length of the last 
      *   dimension corresponds to the dimensionality of the cell descriptor).
      * @param norm_descr The output 1D normalized block descriptor
      * @param block_norm The norm used by the procedure 
      * @param eps The epsilon used for the block normalization 
      *   (to avoid division by zero norm) 
      * @param threshold The threshold used for the block normalization
      *   This is only used with the L2Hys norm, for the clipping of large 
      *   values.
     */ 
    template <typename U, int D>
    void normalizeBlock(const blitz::Array<U,D>& descr,
      blitz::Array<U,1>& norm_descr, const BlockNorm block_norm=L2,
      const double eps=1e-10, const double threshold=0.2)
    {
      // Checks input/output arrays
      int ndescr=1;
      for(int d=0; d<D; ++d) ndescr *= descr.extent(d);
      bob::core::array::assertSameDimensionLength(ndescr, norm_descr.extent(0));

      // Normalizes
      normalizeBlock_(descr, norm_descr, block_norm, eps, threshold);
    }

    /**
      * @brief Abstract class to extract descriptors using a decomposition
      *   into cells (unormalized descriptors) and blocks (groups of cells
      *   used for normalization purpose)
      */
    template <typename T, typename U>
    class BlockCellDescriptors
    {
      public:
        /**
          * Constructor
          */
        BlockCellDescriptors(const size_t height, const size_t width, 
          const size_t cell_dim=8, 
          const size_t cell_y=4, const size_t cell_x=4, 
          const size_t cell_ov_y=0, const size_t cell_ov_x=0,
          const size_t block_y=4, const size_t block_x=4, 
          const size_t block_ov_y=0, const size_t block_ov_x=0);

        /**
          * Destructor
          */
        virtual ~BlockCellDescriptors() {}

        /**
          * Resizes the cache
          */
        void resize(const size_t height, const size_t width);
 
        /**
          * Getters
          */
        inline size_t getHeight() const { return m_height; }
        inline size_t getWidth() const { return m_width; }
        inline size_t getCellDim() const { return m_cell_dim; }
        inline size_t getCellHeight() const { return m_cell_y; }
        inline size_t getCellWidth() const { return m_cell_x; }
        inline size_t getCellOverlapHeight() const { return m_cell_ov_y; }
        inline size_t getCellOverlapWidth() const { return m_cell_ov_x; }
        inline size_t getBlockHeight() const { return m_block_y; }
        inline size_t getBlockWidth() const { return m_block_x; }
        inline size_t getBlockOverlapHeight() const { return m_block_ov_y; }
        inline size_t getBlockOverlapWidth() const { return m_block_ov_x; }
        inline BlockNorm getBlockNorm() const { return m_block_norm; }
        inline double getBlockNormEps() const { return m_block_norm_eps; }
        inline double getBlockNormThreshold() const { return m_block_norm_threshold; }
        /**
          * Setters
          */
        void setHeight(const size_t height)
        { m_height = height; resizeCache(); }
        void setWidth(const size_t width)
        { m_width = width; resizeCache(); }
        void setCellDim(const size_t cell_dim)
        { m_cell_dim = cell_dim; resizeCellCache(); }
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
        void setBlockNorm(const BlockNorm block_norm)
        { m_block_norm = block_norm; }
        void setBlockNormEps(const double block_norm_eps)
        { m_block_norm_eps = block_norm_eps; }
        void setBlockNormThreshold(const double block_norm_threshold)
        { m_block_norm_threshold = block_norm_threshold; }

        /**
          * Disable block normalization. This is performed by setting parameters
          * such that the cells are not further processed, that is
          * block_y=1, block_x=1, block_ov_y=0, block_ov_x=0, and 
          * block_norm=None.
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
        virtual void forward_(const blitz::Array<T,2>& input, 
          blitz::Array<U,3>& output) = 0;
        virtual void forward(const blitz::Array<T,2>& input, 
          blitz::Array<U,3>& output) = 0;

        /**
          * Normalizes all the blocks, given the current state of the cell
          * descriptors
          */
        virtual void normalizeBlocks(blitz::Array<U,3>& output);

      protected:
        // Methods to resize arrays in cache
        virtual void resizeCache();
        virtual void resizeCellCache();
        virtual void resizeBlockCache();

        // Input size
        size_t m_height;
        size_t m_width;
        // Cell-related variables
        size_t m_cell_dim;
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
        BlockNorm m_block_norm;
        double m_block_norm_eps;
        double m_block_norm_threshold;

        // Cache
        // Number of blocks along Y- and X- axes
        size_t m_nb_cells_y;
        size_t m_nb_cells_x;
        size_t m_nb_blocks_y;
        size_t m_nb_blocks_x;

        // Non-normalized descriptors computed at the cell level
        blitz::Array<U,3> m_cell_descriptor;
    };

    template <typename T, typename U>
    BlockCellDescriptors<T,U>::BlockCellDescriptors(const size_t height, 
        const size_t width, const size_t cell_dim, 
        const size_t cell_y, const size_t cell_x, 
        const size_t cell_ov_y, const size_t cell_ov_x,
        const size_t block_y, const size_t block_x, 
        const size_t block_ov_y, const size_t block_ov_x):
      m_height(height), m_width(width),
      m_cell_dim(cell_dim), m_cell_y(cell_y), m_cell_x(cell_x), 
      m_cell_ov_y(cell_ov_y), m_cell_ov_x(cell_ov_x), 
      m_block_y(block_y), m_block_x(block_x), 
      m_block_ov_y(block_ov_y), m_block_ov_x(block_ov_x), 
      m_block_norm(L2), m_block_norm_eps(1e-10), m_block_norm_threshold(0.2) 
    {
      resizeCache();
    }

    template <typename T, typename U>
    void BlockCellDescriptors<T,U>::resize(const size_t height, const size_t width)
    {
      m_height = height;
      m_width = width;
      resizeCache();
    }

    template <typename T, typename U>
    void BlockCellDescriptors<T,U>::resizeCache()
    {
      // Resizes everything else
      resizeCellCache();
    }

    template <typename T, typename U>
    void BlockCellDescriptors<T,U>::resizeCellCache()
    {
      // Resizes the cell-related arrays
      const blitz::TinyVector<int,4> nb_cells = bob::ip::getBlock4DOutputShape(
          m_height, m_width, m_cell_y, m_cell_x, m_cell_ov_y, m_cell_ov_x);
      m_cell_descriptor.resize(nb_cells(0), nb_cells(1), m_cell_dim);
      
      // Updates the class members
      m_nb_cells_y = nb_cells(0);
      m_nb_cells_x = nb_cells(1);
      
      // Number of blocks should be updated
      resizeBlockCache();
    }

    template <typename T, typename U>
    void BlockCellDescriptors<T,U>::resizeBlockCache()
    {
      // Determines the number of blocks per row and column
      blitz::TinyVector<int,4> nb_blocks = ip::getBlock4DOutputShape(
        m_nb_cells_y, m_nb_cells_x, m_block_y, m_block_x, m_block_ov_y, 
        m_block_ov_x);

      // Updates the class members
      m_nb_blocks_y = nb_blocks(0);
      m_nb_blocks_x = nb_blocks(1);
    }

    template <typename T, typename U>
    const blitz::TinyVector<int,3> 
    BlockCellDescriptors<T,U>::getOutputShape() const
    {
      // Returns results
      blitz::TinyVector<int,3> res;
      res(0) = m_nb_blocks_y;
      res(1) = m_nb_blocks_x;
      res(2) = m_block_y * m_block_x * m_cell_dim;
      return res;
    }

    template <typename T, typename U>
    void BlockCellDescriptors<T,U>::disableBlockNormalization()
    {
      m_block_y = 1;
      m_block_x = 1;
      m_block_ov_y = 0;
      m_block_ov_x = 0;
      m_block_norm = None;
      resizeBlockCache();
    }

    template <typename T, typename U>
    void BlockCellDescriptors<T,U>::normalizeBlocks(blitz::Array<U,3>& output)
    { 
      blitz::Range rall = blitz::Range::all();
      // Normalizes by block
      for(size_t by=0; by<m_nb_blocks_y; ++by)
        for(size_t bx=0; bx<m_nb_blocks_x; ++bx)
        {
          blitz::Range ry(by,by+m_block_y-1);
          blitz::Range rx(bx,bx+m_block_x-1);
          blitz::Array<double,3> cells_block = m_cell_descriptor(ry,rx,rall);
          blitz::Array<double,1> block = output(by,bx,rall);
          bob::ip::normalizeBlock_(cells_block, block, m_block_norm, 
            m_block_norm_eps, m_block_norm_threshold);
        }
    }

  }

/**
 * @}
 */
}

#endif /* BOB_IP_BLOCK_CELL_DESCRIPTORS_H */
