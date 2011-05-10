/**
 * @file src/cxx/ip/ip/LBPHSFeatures.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file provides a class to extract features based on histogram
 *   sequences of Local Binary Patterns, as described in:
 */

#ifndef TORCH5SPRO_IP_LBPHS_FEATURES_H
#define TORCH5SPRO_IP_LBPHS_FEATURES_H

#include "core/cast.h"
#include "ip/Exception.h"
#include "ip/block.h"
#include "ip/histo.h"
#include "ip/LBP.h"
#include "ip/LBP4R.h"
#include "ip/LBP8R.h"
#include <list>

namespace Torch {
/**
 * \ingroup libip_api
 * @{
 *
 */
  namespace ip {

  /**
   * @brief This class can be used to extract features based on histogram
   *   sequences of Local Binary Patterns, as described in the following
   *   article:
   */
	class LBPHSFeatures
	{
  	public:

	  	/**
        * @brief Constructor: generates the LBPHSFeatures object
        * @warning Only LBP4R and LBP8R are currently supported
        */
	    LBPHSFeatures( const int block_h, const int block_w, const int overlap_h,
          const int overlap_w, const double lbp_r = 1, const int lbp_p = 8, 
          const bool circular = false, const bool to_average = false, 
          const bool add_average_bit = false, const bool uniform = false, 
          const bool rotation_invariant = false):
        m_lbp(0),
        m_block_h(block_h), m_block_w(block_w), m_overlap_h(overlap_h), 
        m_overlap_w(overlap_w), m_lbp_r(lbp_r), m_lbp_p(lbp_p)
      {
        if( m_lbp_p == 4 )
          m_lbp = new Torch::ip::LBP4R(m_lbp_r, circular, to_average, 
            add_average_bit, uniform, rotation_invariant);
        else if( m_lbp_p == 8 )
          m_lbp = new Torch::ip::LBP8R(m_lbp_r, circular, to_average, 
            add_average_bit, uniform, rotation_invariant);
        else
          throw Torch::ip::LBPUnsupportedNNeighbours(m_lbp_p);
      }

	  	/**
        * @brief Destructor
        */
	    virtual ~LBPHSFeatures() {
        if( m_lbp!=0)
          delete m_lbp;
      }

	  	/**
        * @brief Process a 2D blitz Array/Image by extracting LBPHS features.
        * @param src The 2D input blitz array
        * @param dst A container (with a push_back method such as an STL list)
        *   of 1D uint32_t blitz arrays.
        */
	    template <typename T, typename U> 
      void operator()(const blitz::Array<T,2>& src, U& dst);

      /**
        * @brief Function which returns the number of blocks when applying 
        *   the LBPHSFeatures extractor on a 2D blitz::array/image.
        *   The first dimension is the height (y-axis), whereas the second
        *   one is the width (x-axis).
        * @param src The input blitz array
        */
      template<typename T>
      const int getNBlocks(const blitz::Array<T,2>& src);

	  private:
      /**
        * Attributes
        */
      Torch::ip::LBP *m_lbp;
      int m_block_h;
      int m_block_w;
      int m_overlap_h;
      int m_overlap_w;
      double m_lbp_r;
      int m_lbp_p;
	};

  template <typename T, typename U> 
  void LBPHSFeatures::operator()(const blitz::Array<T,2>& src, 
    U& dst) 
  { 
    // cast to double
    blitz::Array<double,2> double_version = Torch::core::cast<double>(src);

    // get all the blocks
    std::list<blitz::Array<double,2> > blocks;
    blockReference(double_version, blocks, m_block_h, m_block_w, m_overlap_h, m_overlap_w);
  
    // compute an lbp histogram for each block
    for( std::list<blitz::Array<double,2> >::const_iterator it = blocks.begin();
      it != blocks.end(); ++it) 
    {
      // extract lbp using operator()
      blitz::Array<uint16_t,2> lbp_tmp_block(m_lbp->getLBPShape(*it));
      m_lbp->operator()(*it, lbp_tmp_block);

      // Compute the LBP histogram
      blitz::Array<uint64_t, 1> lbp_histo(m_lbp->getMaxLabel());
      histogram<uint16_t>(lbp_tmp_block, lbp_histo, 0, m_lbp->getMaxLabel()-1, 
        m_lbp->getMaxLabel());

      // Push the resulting processed block in the container
      dst.push_back(lbp_histo);
    }
  }

  template<typename T>
  const int LBPHSFeatures::getNBlocks(const blitz::Array<T,2>& src)
  {
    const blitz::TinyVector<int,3> res = getBlockShape(src, m_block_h, 
      m_block_w, m_overlap_h, m_overlap_w); 
    return res(0);
  }


}}

#endif /* TORCH5SPRO_IP_LBPHS_FEATURES_H */
