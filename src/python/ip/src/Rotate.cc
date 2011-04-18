/**
 * @file src/python/ip/src/Rotate.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 * @date Thu 17 Mar 19:12:40 2011 
 *
 * @brief Binds the Rotate class to python
 */

#include <boost/python.hpp>

#include "ip/Rotate.h"

using namespace boost::python;
namespace ip = Torch::ip;

static const char* rotate_doc = "Objects of this class, after configuration, can perform a rotation.";
static const char* angle_to_horizontal_doc = "Get the angle needed to level out (horizontally) two points.";

#define ROTATE_DECL(T) \
  static const blitz::TinyVector<int,2> getOutputShape(const blitz::Array<T,2>& src, const double angle) \
  { \
    return Torch::ip::Rotate::getOutputShape<T>(src,angle); \
  }

ROTATE_DECL(uint8_t)
ROTATE_DECL(uint16_t)
ROTATE_DECL(double)

#define ROTATE_DEF(T) \
  .def("__call__", (void (ip::Rotate::*)(const blitz::Array<T,2>&, blitz::Array<double,2>&))&ip::Rotate::operator()<T>, (arg("self"), arg("input"), arg("output")), "Call an object of this type to perform a rotation of an image.") \
  .def("__call__", (void (ip::Rotate::*)(const blitz::Array<T,2>&, blitz::Array<double,2>&, const double))&ip::Rotate::operator()<T>, (arg("self"), arg("input"), arg("output"), arg("rotation_angle")), "Call an object of this type to perform a rotation of an image with the given angle.") \
  .def("__call__", (void (ip::Rotate::*)(const blitz::Array<T,2>&, const blitz::Array<bool,2>&, blitz::Array<double,2>&, blitz::Array<bool,2>&))&ip::Rotate::operator()<T>, (arg("self"), arg("input"), arg("input_mask"), arg("output"), arg("output_mask")), "Call an object of this type to perform a rotation of an image.") \
  .def("__call__", (void (ip::Rotate::*)(const blitz::Array<T,2>&, const blitz::Array<bool,2>&, blitz::Array<double,2>&, blitz::Array<bool,2>&, const double))&ip::Rotate::operator()<T>, (arg("self"), arg("input"), arg("input_mask"), arg("output"), arg("output_mask"), arg("rotation_angle")), "Call an object of this type to perform a rotation of an image.") \
  .def("getOutputShape", (const blitz::TinyVector<int,2> (*)(const blitz::Array<T,2>&, const double))&getOutputShape, (arg("input"), arg("rotation_angle")), "Return the required output shape for the given input and rotation angle.")

void bind_ip_rotate() {
  enum_<Torch::ip::Rotate::Algorithm>("RotateAlgorithm")
    .value("Shearing", Torch::ip::Rotate::Shearing)
    .value("BilinearInterp", Torch::ip::Rotate::BilinearInterp)
    ;

  class_<ip::Rotate, boost::shared_ptr<ip::Rotate> >("Rotate", rotate_doc, init<const double, optional<const Torch::ip::Rotate::Algorithm> >((arg("rotation_angle"), arg("rotation_algorithm")="Shearing"), "Constructs a Rotate object."))
    .add_property("angle", &ip::Rotate::getAngle, &ip::Rotate::setAngle)
    .add_property("algorithm", &ip::Rotate::getAlgorithm, &ip::Rotate::setAlgorithm)
    ROTATE_DEF(uint8_t)
    ROTATE_DEF(uint16_t)
    ROTATE_DEF(double)
    .staticmethod("getOutputShape")
    ;

  def("getAngleToHorizontal", (const double (*)(const int, const int, const int, const int))&Torch::ip::getAngleToHorizontal, (arg("left_h"), arg("left_w"), arg("right_h"), arg("right_w")), angle_to_horizontal_doc)
    ;
}
