/**
 * @file src/cxx/ip/ip/extrapolateMask.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Extrapolate an image given a mask
 */

#ifndef TORCH5SPRO_IP_EXTRAPOLATE_MASK_H
#define TORCH5SPRO_IP_EXTRAPOLATE_MASK_H

#include <blitz/array.h>
#include "core/array_assert.h"

#include <iostream>

namespace Torch {
/**
 * \ingroup libip_api
 * @{
 *
 */
  namespace ip {

    /**
      * @brief Function which extracts an image with a nearest neighbour 
      *   technique, a boolean mask being given.
      *   a/ The columns of the image are firstly extrapolated wrt. to the
      *   nearest neighbour on the same column.
      *   b/ The rows of the image are the extrapolate wrt. to the
      *   closest neighbour on the same row.
      *   The first dimension is the height (y-axis), whereas the second one
      *   is the width (x-axis).
      * @param src_mask The 2D input blitz array mask.
      * @param img The 2D input/output blitz array/image.
      * @warning The function assumes that the true values on the mask form
      *   a convex area.
      * @warning img is used as both an input and output, in order to provide
      *   high performance. A copy might be done by the user before calling 
      *   the function if required.
      */
    template <typename T> 
    void extrapolateMask( const blitz::Array<bool,2>& src_mask, 
      blitz::Array<T,2>& img)
    {
      // Check input and output size
      Torch::core::array::assertSameShape(src_mask, img);
      Torch::core::array::assertZeroBase(src_mask);
      Torch::core::array::assertZeroBase(img);
      
      // TODO: check that the input mask is convex

      // Determine the "full of false" columns
      blitz::firstIndex i;
      blitz::secondIndex j;

      blitz::Array<bool,1> column_true(blitz::any(src_mask(j,i), j) );
      int true_min_index=blitz::first(column_true);
      int true_max_index=blitz::last(column_true);

      // Extrapolate the "non false" columns
      for(int jj=true_min_index; jj<=true_max_index; ++jj)
      {
        blitz::Array<bool,1> src_col( src_mask( blitz::Range::all(), jj) );
        int i_first = blitz::first(src_col);
        if( i_first>0)
        {
          blitz::Range r_first(0,i_first-1);
          img(r_first,jj) = img(i_first,jj);
        }

        int i_last=blitz::last(src_col);
        if( i_last+1<src_mask.extent(0))  
        {
          blitz::Range r_last(i_last+1,src_mask.extent(0)-1);
          img(r_last,jj) = img(i_last,jj);
        }
      }

      // Extrapolate the rows
      blitz::Range r_left(0,true_min_index-1);
      blitz::Range r_right(true_max_index+1,src_mask.extent(0)-1);
      for(int i=0; i<src_mask.extent(0); ++i)
      {
        if(true_min_index>0)
          img(i,r_left) = img(i,true_min_index);
        if(true_max_index+1<src_mask.extent(1))
          img(i,r_right) = img(i,true_max_index);
      }
    }

  }
/**
 * @}
 */
}

#endif /* TORCH5SPRO_IP_EXTRAPOLATE_MASK_H */
