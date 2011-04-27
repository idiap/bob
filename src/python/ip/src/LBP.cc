/**
 * @file src/python/ip/src/LBP.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 * @date Tue 26 Apr 17:18:40 2011
 *
 * @brief Binds the LBP class to python
 */

#include <boost/python.hpp>

#include <stdint.h>
#include "ip/LBP.h"
#include "ip/LBP4R.h"
#include "ip/LBP8R.h"
#include "ip/LBPTopOperator.h"

using namespace boost::python;
namespace ip = Torch::ip;

static const char* lbp4r_doc = "Objects of this class, after configuration, can compute Local Binary Features using 4 neighbour pixels.";
static const char* lbp8r_doc = "Objects of this class, after configuration, can compute Local Binary Features using 8 neighbour pixels.";

#define LBP4R_CALL_DEF(T) \
  .def("__call__", (void (ip::LBP4R::*)(const blitz::Array<T,2>&, blitz::Array<uint16_t,2>&) const)&ip::LBP4R::operator()<T>, (arg("self"), arg("input"), arg("output")), "Call an object of this type to extract LBP4R features.") \
  .def("__call__", (uint16_t (ip::LBP4R::*)(const blitz::Array<T,2>&, int, int) const)&ip::LBP4R::operator()<T>, (arg("self"), arg("input"), arg("y"), arg("x")), "Call an object of this type to extract LBP4R features.") \
  .def("getLBPShape", (const blitz::TinyVector<int,2> (ip::LBP4R::*)(const blitz::Array<T,2>&) const)&ip::LBP4R::getLBPShape<T>, (arg("self"), arg("input")), "Get the expected size of the output when extracting LBP4R features.")

#define LBP8R_CALL_DEF(T) \
  .def("__call__", (void (ip::LBP8R::*)(const blitz::Array<T,2>&, blitz::Array<uint16_t,2>&) const)&ip::LBP8R::operator()<T>, (arg("self"), arg("input"), arg("output")), "Call an object of this type to extract LBP8R features.") \
  .def("__call__", (uint16_t (ip::LBP8R::*)(const blitz::Array<T,2>&, int, int) const)&ip::LBP8R::operator()<T>, (arg("self"), arg("input"), arg("y"), arg("x")), "Call an object of this type to extract LBP8R features.") \
  .def("getLBPShape", (const blitz::TinyVector<int,2> (ip::LBP8R::*)(const blitz::Array<T,2>&) const)&ip::LBP8R::getLBPShape<T>, (arg("self"), arg("input")), "Get the expected size of the output when extracting LBP8R features.")

#define LBPTOP_CALL_DEF(T) \
  .def("__call__", (void (ip::LBPTopOperator::*)(const blitz::Array<T,3>&, blitz::Array<uint16_t,2>&, blitz::Array<uint16_t,2>&, blitz::Array<uint16_t,2>&) const)&ip::LBPTopOperator::operator()<T>, (arg("self"),arg("input"), arg("xy"), arg("xt"), arg("yt")), "Processes a 3D tensor representing a set of <b>grayscale</b> images and returns (by argument) the three LBP planes calculated. The 3D tensor has to be arranged in this way:\n\n1st dimension => frame height\n2nd dimension => frame width\n4th dimension => time\n\nThe number of frames in the tensor has to be always an odd number. The central frame is taken as the frame where the LBP planes have to be calculated from. The radius in dimension T (3rd dimension) is taken to be (N-1)/2 where N is the number of frames input.")

void bind_ip_lbp_new() {
  class_<ip::LBP, boost::noncopyable>("LBP", "A base class for the LBP-like operators", no_init)
    .add_property("R", &ip::LBP::getRadius, &ip::LBP::setRadius)
    .add_property("P", &ip::LBP::getNNeighbours)
    .add_property("circular", &ip::LBP::getCircular, &ip::LBP::setCircular)
    .add_property("to_average", &ip::LBP::getToAverage, &ip::LBP::setToAverage)
    .add_property("add_average_bit", &ip::LBP::getAddAverageBit, &ip::LBP::setAddAverageBit)
    .add_property("uniform", &ip::LBP::getUniform, &ip::LBP::setUniform)
    .add_property("rotation_invariant", &ip::LBP::getRotationInvariant, &ip::LBP::setRotationInvariant)
    ;

  class_<ip::LBP4R, boost::shared_ptr<ip::LBP4R>, bases<ip::LBP> >("LBP4R", lbp4r_doc, init<optional<const double, const bool, const bool, const bool, const bool, const bool> >((arg("R")=1.0,arg("circular")=false,arg("to_average")=false,arg("add_average_bit")=false,arg("uniform")=false, arg("rotation_invariant")=false), "Construct a new LBP4R object"))
    .add_property("max_label", &ip::LBP4R::getMaxLabel)
    LBP4R_CALL_DEF(uint8_t)
    LBP4R_CALL_DEF(uint16_t)
    LBP4R_CALL_DEF(double)
    ;

  class_<ip::LBP8R, boost::shared_ptr<ip::LBP8R>, bases<ip::LBP> >("LBP8R", lbp8r_doc, init<optional<const double, const bool, const bool, const bool, const bool, const bool> >((arg("R")=1.0,arg("circular")=false,arg("to_average")=false,arg("add_average_bit")=false,arg("uniform")=false, arg("rotation_invariant")=false), "Construct a new LBP8R object"))
    .add_property("max_label", &ip::LBP4R::getMaxLabel)
    LBP8R_CALL_DEF(uint8_t)
    LBP8R_CALL_DEF(uint16_t)
    LBP8R_CALL_DEF(double)
    ;

  class_<ip::LBPTopOperator, boost::shared_ptr<ip::LBPTopOperator> >("LBPTopOperator",
 "Constructs a new LBPTopOperator object starting from the algorithm configuration. Please note this object will always produce rotation invariant 2D codes, also taking into consideration pattern uniformity (u2 variant).\n\nThe radius in X (width) direction is combined with the radius in the Y (height) direction for the calculation of the LBP on the XY (frame) direction. The radius in T is taken from the number of frames input, so it is dependent on the input to LBPTopOperator::operator().\n\nThe current number of points supported in torch is either 8 or 4. Any values differing from that need implementation of specialized functionality.", init<int, int, int, int, int, int>((arg("radius_xy"), arg("points_xy"), arg("radius_xt"), arg("points_xt"),  arg("radius_yt"), arg("points_yt")), "Constructs a new ipLBPTopOperator"))
    LBPTOP_CALL_DEF(uint8_t)
    LBPTOP_CALL_DEF(uint16_t)
    LBPTOP_CALL_DEF(double)
    ;
}
