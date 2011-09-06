/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu 01 Sep 2011 08:45:53 CEST
 *
 * @brief Implementation of Spatio-Temporal Gradients as indicated on the
 * equivalent header file.
 */

#include "ip/SpatioTemporalGradient.h"
#include "sp/convolution.h"
#include "core/array_assert.h"

namespace ip = Torch::ip;
namespace sp = Torch::sp;

void ip::ForwardGradient(const blitz::Array<double,2>& i1,
    const blitz::Array<double,2>& i2, blitz::Array<double,2>& u, 
    blitz::Array<double,2>& v) {

  // all arrays have to have the same shape
  Torch::core::array::assertSameShape(i1, i2);
  Torch::core::array::assertSameShape(u, v);
  Torch::core::array::assertSameShape(i1, u);

  blitz::Array<double,1> kernel1(2);
  kernel1 = +1, -1;
  blitz::Array<double,1> kernel2(2);
  kernel2 = +1, +1;
  blitz::Array<double,2> tmp(i1.shape());

  // [+1 +1] * i1
  sp::convolveSep(i1, kernel2, u, 1, sp::Convolution::Same, 
      sp::Convolution::Mirror);
  // [+1 -1]^T * ([+1 +1]*(i1))
  sp::convolveSep(u, kernel1, u, 0, sp::Convolution::Same,
      sp::Convolution::Mirror);
  
  // [+1 +1] * i2
  sp::convolveSep(i2, kernel2, tmp, 1, sp::Convolution::Same, 
      sp::Convolution::Mirror);
  // [+1 -1]^T * ([+1 +1]*(i2))
  sp::convolveSep(tmp, kernel1, tmp, 0, sp::Convolution::Same,
      sp::Convolution::Mirror);

  u += tmp;
  
  // [-1 +1] * i1
  sp::convolveSep(i1, kernel1, v, 1, sp::Convolution::Same, 
      sp::Convolution::Mirror);
  // [+1 +1]^T([-1 +1]*(i1))
  sp::convolveSep(v, kernel2, v, 0, sp::Convolution::Same,
      sp::Convolution::Mirror);
  
  // [-1 +1] * i2
  sp::convolveSep(i2, kernel1, tmp, 1, sp::Convolution::Same,
      sp::Convolution::Mirror);
  // [+1 +1]^T * ([-1 +1]*(i2))
  sp::convolveSep(tmp, kernel2, tmp, 0, sp::Convolution::Same,
      sp::Convolution::Mirror);
  
  v += tmp;
}

void ip::CentralGradient(const blitz::Array<double,2>& i_prev,
    const blitz::Array<double,2>& i, const blitz::Array<double,2>& i_after,
    blitz::Array<double,2>& u, blitz::Array<double,2>& v) {
  
  // all arrays have to have the same shape
  Torch::core::array::assertSameShape(i_prev, i);
  Torch::core::array::assertSameShape(i_after, i);
  Torch::core::array::assertSameShape(u, v);
  Torch::core::array::assertSameShape(i, u);

  blitz::Array<double,1> kernel1(3);
  kernel1 = +1, 0, -1;
  blitz::Array<double,1> kernel2(3);
  kernel2 = +1, +2, +1;
  blitz::Array<double,2> tmp(i.shape());

  // [+1 +2 +1] * i_prev
  sp::convolveSep(i_prev, kernel2, u, 1, sp::Convolution::Same, 
      sp::Convolution::Mirror);
  // [-1 0 +1]^T * ([+1 +2 +1]*(i_prev))
  sp::convolveSep(u, kernel1, u, 0, sp::Convolution::Same,
      sp::Convolution::Mirror);
  
  // [+1 +2 +1] * i
  sp::convolveSep(i, kernel2, tmp, 1, sp::Convolution::Same, 
      sp::Convolution::Mirror);
  // [-1 0 +1]^T * ([+1 +2 +1]*(i))
  sp::convolveSep(tmp, kernel1, tmp, 0, sp::Convolution::Same,
      sp::Convolution::Mirror);

  u += 2*tmp;
  
  // [+1 +2 +1] * i_after
  sp::convolveSep(i_after, kernel2, tmp, 1, sp::Convolution::Same, 
      sp::Convolution::Mirror);
  // [-1 0 +1]^T * ([+1 +2 +1]*(i_after))
  sp::convolveSep(tmp, kernel1, tmp, 0, sp::Convolution::Same,
      sp::Convolution::Mirror);

  u += tmp;

  // [-1 0 +1] * i_prev
  sp::convolveSep(i_prev, kernel1, v, 1, sp::Convolution::Same,
      sp::Convolution::Mirror);
  // [+1 +2 +1]^T * ([-1 0 +1]*(i_prev))
  sp::convolveSep(v, kernel2, v, 0, sp::Convolution::Same,
      sp::Convolution::Mirror);
  
  // [-1 0 +1] * i
  sp::convolveSep(i, kernel1, tmp, 1, sp::Convolution::Same,
      sp::Convolution::Mirror);
  // [+1 +2 +1]^T([-1 0 +1]*(i))
  sp::convolveSep(tmp, kernel2, tmp, 0, sp::Convolution::Same,
      sp::Convolution::Mirror);
  
  v += 2*tmp;

  // [-1 0 +1] * i_after
  sp::convolveSep(i_after, kernel1, tmp, 1, sp::Convolution::Same,
      sp::Convolution::Mirror);
  // [+1 +2 +1]^T([-1 0 +1]*(i_after))
  sp::convolveSep(tmp, kernel2, tmp, 0, sp::Convolution::Same,
      sp::Convolution::Mirror);
  
  v += tmp;
}
