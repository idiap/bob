/**
 * @file src/cxx/ip/ip/dctFeatureExtract.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file defines a function to dctFeatureExtract a 2D or 3D array/image.
 * The algorithm is strongly inspired by the following article:
 * 
 */

#ifndef TORCH5SPRO_IP_ZIGZAG_H
#define TORCH5SPRO_IP_ZIGZAG_H

#include "core/logging.h"
#include "ip/Exception.h"
#include "ip/common.h"

namespace Torch {
/**
 * \ingroup libip_api
 * @{
 *
 */
  namespace ip {

	  namespace detail {
	  }
	  
    	  template<typename T, typename U>
	  void zigzag(const blitz::Array<T,2>& src, blitz::Array<T,1>& dst, int n_dct_coefficients_kept = -1)
	  {
		  // Checks that the src array has zero base indices
		  detail::assertZeroBase( src);

		  // we can currently only handle up to the major diagonal
		  const int size = src.extent(0);
		  if( n_dct_coefficients_kept >= size * (size - 1) / 2 )
			  throw ParamOutOfBoundaryError("n_dct_coefficients_kept", false, n_dct_coefficients_kept, 0);

		  // 
		  if (-1 == n_dct_coefficients_kept) 
			  n_dct_coefficients_kept = src.extent(0) * src.extent(1);

		  //
		  int current_diagonal         = 0;
		  int diagonal_left_to_right_p = true;
		  int diagonal_index           = 0;
		  for (int iii = 0; iii < n_dct_coefficients_kept; ++iii ) {
			  int x, y;
			  
			  if (diagonal_left_to_right_p) {
				  x = diagonal_index;
				  y = current_diagonal - x;
			  } else {
				  y = diagonal_index;
				  x = current_diagonal - y;
			  }

			  if (current_diagonal <= diagonal_index) {
				  ++current_diagonal;
				  diagonal_left_to_right = !diagonal_left_to_right;
				  diagonal_index = 0; 
			  }  else {
				  ++diagonal_index;
			  }
		  }
	  }
  }

/**
 * @}
 */
}

#endif /* TORCH5SPRO_IP_ZIGZAG_H */
