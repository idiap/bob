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

    	  template<typename T>
	  void zigzag(const blitz::Array<T,2>& src, blitz::Array<T,1>& dst, int n_dct_coefficients_kept = -1)
	  {
		  // the maximum number of dct coeff:s that we can handle rhight now
		  const int size = src.extent(0);
		  const int max_n_dct = size * (size - 1) / 2;
		  
		  // Checks that the src array has zero base indices
		  detail::assertZeroBase( src);

		  // we can currently only handle up to the major diagonal
		  if( n_dct_coefficients_kept > max_n_dct )
			  throw ParamOutOfBoundaryError("n_dct_coefficients_kept", false, n_dct_coefficients_kept, 0);

		  // make sure that destination is of correct size
		  if( dst.extent(0) != n_dct_coefficients_kept ) 
			  dst.resize( n_dct_coefficients_kept );

		  // 
		  if (-1 == n_dct_coefficients_kept) 
			  n_dct_coefficients_kept = max_n_dct;

		  //
		  int current_diagonal         = 0;
		  int diagonal_left_to_right_p = false;
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

			  // save the value
			  dst(iii) = src(x, y);

			  if (current_diagonal <= diagonal_index) {
				  ++current_diagonal;
				  diagonal_left_to_right_p = !diagonal_left_to_right_p;
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
