/**
 * @file src/cxx/ip/src/shift.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 */

#include "ip/shift.h"
#include "ip/Exception.h"

namespace ipd = Torch::ip::detail;

void ipd::shiftParameterCheck( const int shift_y, const int shift_x,
  const int src_height, const int src_width)
{
  // Check parameters and throw exception if required
  if( shift_y <= -src_height ) {
    throw ParamOutOfBoundaryError("shift_y", false, shift_y, 
      -src_height+1);
  }
  if( shift_x <= -src_width ) {
    throw ParamOutOfBoundaryError("shift_x", false, shift_x, 
      -src_width+1);
  }
  if( shift_y >= src_height ) {
    throw ParamOutOfBoundaryError("shift_y", true, shift_y, 
      src_height-1);
  }
  if( shift_x >= src_width ) {
    throw ParamOutOfBoundaryError("shift_x", true, shift_x, 
      src_width-1);
  }
}

