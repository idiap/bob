/**
 * @file src/cxx/ip/src/crop.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 */

#include "ip/crop.h"

namespace ipd = Torch::ip::detail;

void ipd::cropParameterCheck( const int crop_y, const int crop_x,
  const int crop_h, const int crop_w, const int src_height, 
  const int src_width)
{
  // Check parameters and throw exception if required
  if( crop_y<0) {
    throw ParamOutOfBoundaryError("crop_y", false, crop_y, 0);
  }
  if( crop_x<0 ) {
    throw ParamOutOfBoundaryError("crop_x", false, crop_x, 0);
  }
  if( crop_h<0) {
    throw ParamOutOfBoundaryError("crop_h", false, crop_h, 0);
  }
  if( crop_w<0) {
    throw ParamOutOfBoundaryError("crop_w", false, crop_w, 0);
  }
  if( crop_y+crop_h>src_height ) {
    throw ParamOutOfBoundaryError("crop_y+crop_h", true, crop_y+crop_h, 
      src_height );
  }
  if( crop_x+crop_w>src_width ) {
    throw ParamOutOfBoundaryError("crop_x+crop_w", true, crop_x+crop_w, 
      src_width );
  }
}

