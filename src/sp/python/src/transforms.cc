/**
 * @file src/python/ip/color.cc 
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Binds Color to python 
 */

#include <boost/python.hpp>

#include "core/Tensor.h"
#include "sp/spCore.h"
#include "sp/spFCT.h"
#include "sp/spDCT_naive.h"
#include "sp/spFCT_oourafft.h"
#include "sp/spFFT_oourafft.h"
#include "sp/spFFT_pack41.h"
#include "sp/spDFT.h"

using namespace boost::python;

void bind_sp_transforms()
{
  class_<Torch::spDCT_naive, bases<Torch::spCore> >("spDCT_naive", "This class is designed to perform a DCT (or inverse DCT) over an input tensor. The result is a tensor of the same storage type.", init<optional<bool> >(arg("inverse"), "Creates an new DCT operator."))
    ;
  class_<Torch::spFCT, bases<Torch::spCore> >("spFCT", "This class is designed to perform a FCT (or inverse FCT) over an input tensor. The result is a tensor of the same storage type.", init<optional<bool> >(arg("inverse"), "Creates an new FCT operator."))
    ;
  class_<Torch::spFCT_oourafft, bases<Torch::spCore> >("spFCT_oourafft", "This class is designed to perform a FCT (or inverse FCT) over an input tensor. The result is a tensor of the same storage type.", init<optional<bool> >(arg("inverse"), "Creates an new FCT operator."))
    ;
  class_<Torch::spFFT_oourafft, bases<Torch::spCore> >("spFFT_oourafft", "This class is designed to perform an FFT (or inverse FFT) over an input tensor. The result is a tensor of the same storage type.", init<optional<bool> >(arg("inverse"), "Creates an new FFT operator."))
    ;
  class_<Torch::spFFT_pack41, bases<Torch::spCore> >("spFFT_pack41", "This class is designed to perform an FFT (or inverse FFT) over an input tensor. The result is a tensor of the same storage type.", init<optional<bool> >(arg("inverse"), "Creates an new FFT operator."))
    ;
  class_<Torch::spDFT, bases<Torch::spCore> >("spDFT", "This class is designed to perform a DFT (or inverse DFT) over an input tensor. The result is a tensor of the same storage type.", init<optional<bool> >(arg("inverse"), "Creates a new DFT operator."))
    ;
}
