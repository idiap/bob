/**
 * @file src/cxx/ip/ip/DCTFeatures.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file provides a class to extract DCT features as described in:
 *   "Polynomial Features for Robust Face Authentication", 
 *   from C. Sanderson and K. Paliwal, in the proceedings of the 
 *   IEEE International Conference on Image Processing 2002.
 */

#ifndef TORCH5SPRO_IP_DCT_FEATURES_H
#define TORCH5SPRO_IP_DCT_FEATURES_H

#include "core/cast.h"
#include "ip/Exception.h"

#include <list>

#include "ip/block.h"
#include "sp/DCT2D.h"
#include "ip/zigzag.h"

namespace Torch {
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
  */
	class DCTFeatures
	{
  	public:

	  	/**
        * @brief Constructor: generates the Difference of Gaussians filter
        */
	    DCTFeatures( const int block_h, const int block_w, const int overlap_h, 
        const int overlap_w, const int n_dct_coefs): m_dct2d(0),
          m_block_h(block_h), m_block_w(block_w), m_overlap_h(overlap_h), 
          m_overlap_w(overlap_w), m_n_dct_coefs(n_dct_coefs)
      {
        m_dct2d = new Torch::sp::DCT2D(block_h, block_w);
      }

	  	/**
        * @brief Destructor
        */
	    virtual ~DCTFeatures() {
        if( m_dct2d!=0)
          delete m_dct2d;
      }

	  	/**
        * @brief Process a 2D blitz Array/Image by extracting DCT features.
        * @param src The 2D input blitz array
        * @param dst A container (with a push_back method such as an STL list)
        *   of 1D double blitz arrays.
        */
	    template <typename T, typename U> 
      void operator()(const blitz::Array<T,2>& src, U& dst);

	  private:
      /**
        * Attributes
        */
      Torch::sp::DCT2D *m_dct2d;
      int m_block_h;
      int m_block_w;
      int m_overlap_h;
      int m_overlap_w;
      int m_n_dct_coefs;
	};

  template <typename T, typename U> 
  void DCTFeatures::operator()(const blitz::Array<T,2>& src, 
    U& dst) 
  { 
    // cast to double
    blitz::Array<double,2> double_version = Torch::core::cast<double>(src);

    // get all the blocks
    std::list<blitz::Array<double,2> > blocks;
    blockReference(double_version, blocks, m_block_h, m_block_w, m_overlap_h, 
      m_overlap_w);
  
    /// dct extract each block
    for( std::list<blitz::Array<double,2> >::const_iterator it = blocks.begin(); 
      it != blocks.end(); ++it) 
    {
      // extract dct using operator()
      blitz::Array<double,2> dct_tmp_block(m_block_h, m_block_w);
      m_dct2d->operator()(*it, dct_tmp_block);

      // extract the required number of coefficients using the zigzag pattern
      blitz::Array<double,1> dct_block_zigzag(m_n_dct_coefs);
      zigzag(dct_tmp_block, dct_block_zigzag, m_n_dct_coefs);
      
      // Push the resulting processed block in the container
      dst.push_back(dct_block_zigzag);
    }
  }

}}

#endif /* TORCH5SPRO_IP_DCT_FEATURES_H */
