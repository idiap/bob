/**
 * @file ip/python/src/filters.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief All IP filters in one shot
 */

#include <boost/python.hpp>

//#include "ip/ipCrop.h"
//#include "ip/ipFlip.h"
//#include "ip/ipHisto.h"
#include "ip/ipHistoEqual.h"
//#include "ip/ipIntegral.h"
#include "ip/ipMSRSQIGaussian.h"
#include "ip/ipMultiscaleRetinex.h"
#include "ip/ipRelaxation.h"
//#include "ip/ipRescaleGray.h"
//#include "ip/ipRotate.h"
//#include "ip/ipScaleYX.h"
#include "ip/ipSelfQuotientImage.h"
//#include "ip/ipShift.h"
#include "ip/ipSmoothGaussian.h"
#include "ip/ipSobel.h"
//#include "ip/ipTanTriggs.h"
#include "ip/ipVcycle.h"

using namespace boost::python;

void bind_ip_filters()
{
//  class_<Torch::ipCrop, bases<Torch::ipCore> >("ipCrop", "A filter to crop images. Paramters (x,y,w,h). The result is a tensor of the same storage type.", init<>());
//  class_<Torch::ipFlip, bases<Torch::ipCore> >("ipFlip", "A filter to flip images. Parameters (vertical). The result is a tensor of the same storage type.", init<>());
//  class_<Torch::ipHisto, bases<Torch::ipCore> >("ipHisto", "This class is designed to compute the histogram of some Image (3D ShortTensor). The result is 2D IntTensor with the dimensions (bin counts, planes)", init<>());
  class_<Torch::ipHistoEqual, bases<Torch::ipCore> >("ipHistoEqual", "This class is designed to enhance an image using histogram equalisation. The output is the normalized image.", init<>());
//  class_<Torch::ipIntegral, bases<Torch::ipCore> >("ipIntegral", "This class is designed to compute the integral image of any 2D/3D tensor type like (height x width [x color channels/modes]). The result will have the same dimensions and size/dimension as the input, but the input type will vary like: Char => Int; Short => Int; Int => Int; Long => Long; Float => Double; Double => Double. NB: For a 3D tensor, the integral image is computed for each 3D channel (that is the third dimension -> e.g. color channels).",  init<>());
  class_<Torch::ipMSRSQIGaussian, bases<Torch::ipCore> >("ipMSRSQIGaussian", "This class is designed to apply a normalize Gaussian or Weighed Gaussian filtering to an image (3D ShortTensor) by convolving a NxM Gaussian or Weighed Gaussianfilter. The implementation is described in: 'Face Recognition under Varying Lighting conditions using Self Quotient Image from Wang, Li and Wang, 2004' for the SelfQuotientImage. In particular, it performs: 1. Normalization by the area of the filter; 2. Mirror interpolation at the border. The result will be a 3D ShortTensor image having the same number of planes. Parameters (RadiusX, RadiusY, Sigma, Weighed).", init<>());
  class_<Torch::ipMultiscaleRetinex, bases<Torch::ipCore> >("ipMultiscaleRetinex", "This class is designed to perform the Multiscale Retinex algorithm on an image.", init<>());
  class_<Torch::ipRelaxation, bases<Torch::ipCore> >("ipRelaxation", "This class is designed to apply relaxation on the linear system induced by the discretization of an elliptic PDE (diffusion). Relaxation is an iterative method allowing the resolution (approximation) of large and sparse linear systems. Here, the Gauss-Seidel scheme with red-black ordering is used (see multigrid.h file). The number of relaxation steps can be provided (default 10).", init<>());
//  class_<Torch::ipRescaleGray, bases<Torch::ipCore> >("ipRescaleGray", "This class is designed to rescale any Tensor into a 'short' typed image (0 to 255). The result is a short tensor.", init<>());
//  class_<Torch::ipRotate, bases<Torch::ipCore> >("ipRotate", "This class is designed to rotate an image. The result is a tensor of the same storage type, but may have a different size. It implements the rotation by shear also called Paeth rotation ('A fast algorithm for general raster rotation, by Paeth'). Parameters (centerx, centery, angle).", init<>());
//  class_<Torch::ipScaleYX, bases<Torch::ipCore> >("ipScaleYX", "This class is designed to scale an image.	The result is a tensor of the same storage type. Parameters (width, height).", init<>());
  class_<Torch::ipSelfQuotientImage, bases<Torch::ipCore> >("ipSelfQuotientImage", "This class is designed to perform the Self Quotient Image algorithm on an image.", init<>());
//  class_<Torch::ipShift, bases<Torch::ipCore> >("ipShift", "This class is designed to shift an image.	The result is a tensor of the same storage type and the same size. Parameters (shiftx, shifty).", init<>());
  class_<Torch::ipSmoothGaussian, bases<Torch::ipCore> >("ipSmoothGaussian", "This class is designed to smooth an image (3D ShortTensor) by convolving a NxM Gaussian filter.	The result will be a 3D ShortTensor image having the same number of planes. Parameters (RadiusX, RadiusY, Sigma).", init<>());
  class_<Torch::ipSobel, bases<Torch::ipCore> >("ipSobel", "This class is designed to convolve a sobel mask with an image. The result is 3 tensors of the INT storage type: the Ox gradient, the Oy gradient and the edge magnitude.", init<>());
//  class_<Torch::ipTanTriggs, bases<Torch::ipCore> >("ipTanTriggs", "This class is designed to perform the preprocessing chain of Tan and Triggs to normalize images as described in 'Enhanced Local Feature Sets for Face Recognition Under Difficult Lighting Conditions by Tan and Triggs'.", init<>());
  class_<Torch::ipVcycle, bases<Torch::ipCore> >("ipVcycle", "This class implements the multigrid V-cycle algorithm. The V-cycle algorithm is an iterative method allowing the resolution (approximation) of large and sparse linear systems induced by partial differential equations. Multiple grids of different resolution are used in order to speed up the resolution. Note that the number of grids is dependent on the initial size of the 2D data. This algorithm is described on 'A Multigrid Tutorial, by Briggs'.", init<>());
}
