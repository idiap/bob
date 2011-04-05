/**
 * @file src/cxx/ip/ip/dctFeatureExtract.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file defines a function to dctFeatureExtract a 2D or 3D array/image.
 * 
 */

#ifndef TORCH5SPRO_IP_DCTFEATUREEXTRACT_H
#define TORCH5SPRO_IP_DCTFEATUREEXTRACT_H

#include "core/logging.h"
#include "ip/Exception.h"
#include "ip/common.h"
#include "sp/FCT.h"

namespace Torch {
/**
 * \ingroup libip_api
 * @{
 *
 */
  namespace ip {

    namespace detail {
    }

	  /** 
	   * @brief Function which extract DCT features from a 2D blitz::array/image of a given type.
	   */

	  template<typename T, typename U>
	  void dctFeatureExtract(const blitz::Array<T,2>& src, U& dst, 
				 const int block_w, const int block_h, const int overlap_w, 
				 const int overlap_h) 
	  {
		  // get all the blocks
		  std::list<blitz::Array<T, 2> > blocks;
		  block(src, blocks, block_w, block_h, overlap_w, overlap_h);
	
		  // create a dct extractor 
		  DCT2D dct(block_w, block_h);

		  // preallocate two arrays.
		  blitz::Array<double, 2> double_version(block_w, block_h);
		  blitz::Array<T, 2> dct_tmp_block;

		  /// dct extract each block
		  for (U::const_iterator it = dst.begin(); it != dst.end(); ++it) {

			  // dct require double type, therefore cast it
			  double_version = Torch::core::cast<double>(*it);

			  // extract dct using operator()
			  dct(double_version, dct_tmp_block);

			  // create a copy of the tmp_block.
			  dst.push_back(dct_tmp_block);
		  }
	  }
  }

/**
 * @}
 */
}

#endif /* TORCH5SPRO_IP_DCTFEATUREEXTRACT_H */
