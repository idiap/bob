/**
 * @file src/python/ip/color.cc 
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Binds Color to python 
 */

#include <boost/python.hpp>

#include "core/Tensor.h"
#include "sp/spCore.h"
#include "sp/spDCT.h"
#include "sp/spDCT_pack41.h"
#include "sp/spFFT.h"
#include "sp/spFFT_pack41.h"

using namespace boost::python;

void bind_sp_transforms()
{
  class_<Torch::spDCT, bases<Torch::spCore> >("spDCT", "This class is designed to perform a DCT (or inverse DCT) over an input tensor. The result is a tensor of the same storage type.", init<optional<bool> >(arg("inverse"), "Creates an new DCT operator."))
    ;
  class_<Torch::spDCT_pack41, bases<Torch::spCore> >("spDCT_pack41", "This class is designed to perform a DCT (or inverse DCT) over an input tensor. The result is a tensor of the same storage type.", init<optional<bool> >(arg("inverse"), "Creates an new DCT operator."))
    ;
  class_<Torch::spFFT, bases<Torch::spCore> >("spFFT", "This class is designed to perform an FFT (or inverse FFT) over an input tensor. The result is a tensor of the same storage type.", init<optional<bool> >(arg("inverse"), "Creates an new FFT operator."))
    ;
  class_<Torch::spFFT_pack41, bases<Torch::spCore> >("spFFT_pack41", "This class is designed to perform an FFT (or inverse FFT) over an input tensor. The result is a tensor of the same storage type.", init<optional<bool> >(arg("inverse"), "Creates an new FFT operator."))
    ;
}
