/**
 * @file ip/python/src/filters.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief All IP filters in one shot
 */

#include <boost/python.hpp>

#include "ip/ipHistoEqual.h"
#include "ip/ipMSRSQIGaussian.h"
#include "ip/ipMultiscaleRetinex.h"
#include "ip/ipRelaxation.h"
#include "ip/ipSelfQuotientImage.h"
#include "ip/ipSmoothGaussian.h"
#include "ip/ipSobel.h"
#include "ip/ipVcycle.h"

using namespace boost::python;

void bind_ip_filters()
{
  class_<Torch::ipHistoEqual, bases<Torch::ipCore> >("ipHistoEqual", "This class is designed to enhance an image using histogram equalisation. The output is the normalized image.", init<>());
  class_<Torch::ipMSRSQIGaussian, bases<Torch::ipCore> >("ipMSRSQIGaussian", "This class is designed to apply a normalize Gaussian or Weighed Gaussian filtering to an image (3D ShortTensor) by convolving a NxM Gaussian or Weighed Gaussianfilter. The implementation is described in: 'Face Recognition under Varying Lighting conditions using Self Quotient Image from Wang, Li and Wang, 2004' for the SelfQuotientImage. In particular, it performs: 1. Normalization by the area of the filter; 2. Mirror interpolation at the border. The result will be a 3D ShortTensor image having the same number of planes. Parameters (RadiusX, RadiusY, Sigma, Weighed).", init<>());
  class_<Torch::ipMultiscaleRetinex, bases<Torch::ipCore> >("ipMultiscaleRetinex", "This class is designed to perform the Multiscale Retinex algorithm on an image.", init<>());
  class_<Torch::ipRelaxation, bases<Torch::ipCore> >("ipRelaxation", "This class is designed to apply relaxation on the linear system induced by the discretization of an elliptic PDE (diffusion). Relaxation is an iterative method allowing the resolution (approximation) of large and sparse linear systems. Here, the Gauss-Seidel scheme with red-black ordering is used (see multigrid.h file). The number of relaxation steps can be provided (default 10).", init<>());
  class_<Torch::ipSelfQuotientImage, bases<Torch::ipCore> >("ipSelfQuotientImage", "This class is designed to perform the Self Quotient Image algorithm on an image.", init<>());
  class_<Torch::ipSmoothGaussian, bases<Torch::ipCore> >("ipSmoothGaussian", "This class is designed to smooth an image (3D ShortTensor) by convolving a NxM Gaussian filter.	The result will be a 3D ShortTensor image having the same number of planes. Parameters (RadiusX, RadiusY, Sigma).", init<>());
  class_<Torch::ipSobel, bases<Torch::ipCore> >("ipSobel", "This class is designed to convolve a sobel mask with an image. The result is 3 tensors of the INT storage type: the Ox gradient, the Oy gradient and the edge magnitude.", init<>());
  class_<Torch::ipVcycle, bases<Torch::ipCore> >("ipVcycle", "This class implements the multigrid V-cycle algorithm. The V-cycle algorithm is an iterative method allowing the resolution (approximation) of large and sparse linear systems induced by partial differential equations. Multiple grids of different resolution are used in order to speed up the resolution. Note that the number of grids is dependent on the initial size of the 2D data. This algorithm is described on 'A Multigrid Tutorial, by Briggs'.", init<>());
}
