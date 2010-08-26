/**
 * @file python/src/ip/lbp.c
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Binds all LBP constructions into python
 */

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>

#include "ip/ipLBP.h"
#include "ip/ipLBP8R.h"
#include "ip/ipLBP4R.h"
#include "ip/ipLBPTopOperator.h"

using namespace boost::python;

static boost::shared_ptr<Torch::Image> lbp_batch(Torch::ipLBP& op, const Torch::Image& i) {
  const int w = i.getWidth();
  const int h = i.getHeight();
  const int max_lbp = op.getMaxLabel();
  const float inv_max_lbp = 255.0f / (max_lbp + 0.0f);
  boost::shared_ptr<Torch::Image> retval(new Torch::Image(w, h, 1));
  bool success = true;
  for (int x = 1; x < w - 1; x ++)
    for (int y = 1; y < h - 1; y ++) {
      success &= op.setXY(x, y);
      success &= op.process(i);
      (*retval)(y, x, 0) = (short)(inv_max_lbp * op.getLBP() + 0.5f);
    }
  if (success) return retval;
  return boost::shared_ptr<Torch::Image>();
}

void bind_ip_lbp()
{
  class_<Torch::ipLBP, bases<Torch::ipCore>, boost::noncopyable>("ipLBP", 
      "This class computes the LBP code at a given location in the image", 
      no_init)
    .def("setXY", &Torch::ipLBP::setXY, (arg("self"), arg("x"), arg("y")), "Sets the LBP location")
    .add_property("r", &Torch::ipLBP::getR, &Torch::ipLBP::setR)
    .add_property("x", &Torch::ipLBP::getX)
    .add_property("y", &Torch::ipLBP::getY)
    .add_property("value", &Torch::ipLBP::getLBP)
    .add_property("max_label", &Torch::ipLBP::getMaxLabel)
    .def("batch", &lbp_batch, (arg("self"), arg("image")), "Processes the whole image given as input and return a gray-scaled image with the same size. This handle avoids the normal usage loop for the simplest use-case.")
    ;

  class_<Torch::ipLBP4R, bases<Torch::ipLBP> >("ipLBP4R",
      "This class implements LBP4R operators, where 'r' is the radius. Uses the 'Uniform' and 'RotInvariant' boolean options.", no_init)
    .def(init<optional<int> >((arg("r")=1), "Builds a new ipLBP4R object with the given radius."))
    ;

  class_<Torch::ipLBP8R, bases<Torch::ipLBP> >("ipLBP8R",
      "This class implements LBP8R operators, where 'r' is the radius. Uses the 'Uniform' and 'RotInvariant' boolean options.", no_init)
    .def(init<optional<int> >((arg("r")=1), "Builds a new ipLBP8R object with the given radius."))
    ;

  class_<Torch::ipLBPTopOperator, bases<Torch::Object> >("ipLBPTopOperator",
 "Constructs a new ipLBPTopOperator object starting from the algorithm configuration. Please note this object will always produce rotation invariant 2D codes, also taking into consideration pattern uniformity (u2 variant).\n\nThe radius in X (width) direction is combied with the radius in the Y (height) direction for the calculation of the LBP on the XY (frame) direction. The radius in T is taken from the number of frames input, so it is dependent on the input to ipLBPTopOperator::process().\n\nAll input parameters are changeable throught the Torch::Object interface, following the same nomenclature as for the variables in this constructor.\n\nThe current number of points supported in torch is either 8 or 4. Any values differing from that need implementation of specialized functionality.", no_init)
    .def(init<int, int, int, int, int, int>((arg("radius_xy"), arg("points_xy"), arg("radius_xt"), arg("points_xt"),  arg("radius_yt"), arg("points_yt")), "Constructs a new ipLBPTopOperator"))
    .def("process", &Torch::ipLBPTopOperator::process, (arg("input"), arg("xy"), arg("xt"), arg("yt")), "Processes a 4D tensor representing a set of <b>grayscale</b> images and returns (by argument) the three LBP planes calculated. The 4D tensor has to be arranged in this way:\n\n1st dimension => frame height\n2nd dimension => frame width\n3rd dimension => grayscale frame values\n4th dimension => time\n\nThe number of frames in the tensor has to be always an odd number. The central frame is taken as the frame where the LBP planes have to be calculated from. The radius in dimension T (4th dimension) is taken to be (N-1)/2 where N is the number of frames input.")
    ;
}
