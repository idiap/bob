/**
 * @file src/python/ip/color.cc 
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Binds Color to python 
 */

#include <boost/python.hpp>

#include "core/Tensor.h"
#include "core/spCore.h"
#include "sp/spDCT.h"
#include "sp/spFFT.h"
#include "ip/Image.h"

using namespace boost::python;

/**
 * Processes an input image by selecting a plane and slicing the input
 * tensor. Places the output on the output image
 *
 * @param op The operator
 * @param input The input image
 * @param output The output image
 * @param plane To which color plane apply the operator
 * 
 */
static bool dct2d_image(Torch::spCore& op, const Torch::Image& input,
                        Torch::Image& output, int plane=0) {
  Torch::ShortTensor tmp;
  tmp.select(&input, 2, plane);
  Torch::FloatTensor finput(input.size(0), input.size(1));
  finput.copy(&tmp);
  if (!op.process(finput)) return false;
  const Torch::Tensor& ot = op.getOutput(0);
  if (ot.size(1) != output.getWidth() || ot.size(0) != output.getHeight()) {
    output.resize(ot.size(1), ot.size(0), 1);
  }
  Torch::ShortTensor sel;
  sel.select(&output, 2, plane);
  sel.copy(&ot); //this should copy the output into the right planes
  return true;
}

/**
 * Processes an input image by selecting a plane and slicing the input
 * tensor. Places the output on the output image
 *
 * @param op The operator
 * @param input The input image
 * @param output The output image
 * @param plane To which color plane apply the operator
 * 
 */
static bool fft2d_image(Torch::spCore& op, const Torch::Image& input,
                        Torch::Image& output, int plane=0) {
  Torch::ShortTensor tmp;
  tmp.select(&input, 2, plane);
  Torch::FloatTensor finput(input.size(0), input.size(1));
  finput.copy(&tmp);
  if (!op.process(finput)) return false;
  const Torch::Tensor& ot = op.getOutput(0);
  if (ot.size(1) != output.getWidth() || ot.size(0) != output.getHeight()) {
    output.resize(ot.size(1), ot.size(0), 2);
  }
  output.copy(&ot);
  return true;
}

BOOST_PYTHON_FUNCTION_OVERLOADS(dct2d_image_overloads, dct2d_image, 3, 4);
BOOST_PYTHON_FUNCTION_OVERLOADS(fft2d_image_overloads, fft2d_image, 3, 4);

void bind_sp_transforms()
{
  class_<Torch::spDCT, bases<Torch::spCore> >("spDCT", "This class is designed to perform a DCT (or inverse DCT) over an input tensor. The result is a tensor of the same storage type.", init<optional<bool> >(arg("inverse"), "Creates an new DCT operator."))
    .def("processImage", &dct2d_image, dct2d_image_overloads((arg("self"), arg("input"), arg("output"), arg("plane")=0), "Just like process(), but assumes the input is an image and the output is another image. This method will accept color images as input, but you must select to which plane to apply the operation."))
    ;
  class_<Torch::spFFT, bases<Torch::spCore> >("spFFT", "This class is designed to perform an FFT (or inverse FFT) over an input tensor. The result is a tensor of the same storage type.", init<optional<bool> >(arg("inverse"), "Creates an new FFT operator."))
    .def("processImage", &fft2d_image, fft2d_image_overloads((arg("self"), arg("input"), arg("output"), arg("plane")=0), "Just like process(), but assumes the input is an image and the output is another image. This method will accept color images as input, but you must select to which plane to apply the operation."))
    ;
}
