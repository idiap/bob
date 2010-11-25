/**
 * @file src/sp/python/src/transforms.cc 
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Binds SP transforms to python 
 */

#include <boost/python.hpp>

#include "core/Tensor.h"
#include "sp/spCore.h"
#include "sp/spDCT.h"
#include "sp/spFCT.h"
#include "sp/spFCT_oourafft.h"
#include "sp/spDFT.h"
#include "sp/spFFT.h"
#include "sp/spFFT_oourafft.h"

using namespace boost::python;

void bind_sp_transforms()
{
  class_<Torch::spDCT, bases<Torch::spCore> >("spDCT", "This class is designed to perform a Discrete Cosine Transform (DCT) (or inverse DCT) over an input tensor using a naive algorithm. The output is a FloatTensor. A more efficient alternative is to use spFCT.", init<optional<bool> >(arg("inverse"), "Creates a new DCT operator."))
    ;
  class_<Torch::spFCT, bases<Torch::spCore> >("spFCT", "This class is designed to perform a Fast Cosine Transform (FCT) (or inverse FCT) using the FFTPACK 4.1 library over an input tensor. The ouput is a FloatTensor, and is supposed to be identical to the output of spDCT.", init<optional<bool> >(arg("inverse"), "Creates a new FCT operator."))
    ;
  class_<Torch::spFCT_oourafft, bases<Torch::spCore> >("spFCT_oourafft", "This class is designed to perform a Fast Cosine Transform (FCT) (or inverse FCT) over an input tensor using the oourafft library. It only allows power of two length. The output is a FloatTensor, and is supposed to be identical to the output of spDCT.", init<optional<bool> >(arg("inverse"), "Creates a new FCT operator."))
    ;
  class_<Torch::spDFT, bases<Torch::spCore> >("spDFT", "This class is designed to perform a Discrete Fourier Transform (DFT) (or inverse DFT) over an input tensor using a naive algorithm. The output is a FloatTensor. A more efficient alternative is to used spFFT.", init<optional<bool> >(arg("inverse"), "Creates a new DFT operator."))
    ;
  class_<Torch::spFFT, bases<Torch::spCore> >("spFFT", "This class is designed to perform a Fast Fourier Transform (FFT) (or inverse FFT) over an input tensor using the FFTPACK 4.1 library. The output is a FloatTensor, and is supposed to be identical to the output of spDFT.", init<optional<bool> >(arg("inverse"), "Creates a new FFT operator."))
    ;
  class_<Torch::spFFT_oourafft, bases<Torch::spCore> >("spFFT_oourafft", "This class is designed to perform a Fast Fourier Transform (FFT) (or inverse FFT) over an input tensor using the oourafft library. It only allows power of two length. The output is a FloatTensor, and is supposed to be identical to the output of spDFT.", init<optional<bool> >(arg("inverse"), "Creates an new FFT operator."))
    ;
}
