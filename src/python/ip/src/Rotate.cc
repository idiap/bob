/**
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @date Thu 17 Mar 19:12:40 2011 
 *
 * @brief Binds the Rotate class to python
 */


#include "ip/Rotate.h"
#include "core/python/ndarray.h"

using namespace boost::python;
namespace ip = Torch::ip;
namespace tp = Torch::python;
namespace ca = Torch::core::array;

static const char* rotate_doc = "Objects of this class, after configuration, can perform a rotation.";
static const char* angle_to_horizontal_doc = "Get the angle needed to level out (horizontally) two points.";

static object getOutputShape (tp::const_ndarray src, double a) {
  const ca::typeinfo& info = src.type();
  switch (info.dtype) {
    case ca::t_uint8: 
      return object(ip::Rotate::getOutputShape<uint8_t>(src.bz<uint8_t,2>(), a));
    case ca::t_uint16:
      return object(ip::Rotate::getOutputShape<uint16_t>(src.bz<uint16_t,2>(), a));
    case ca::t_float64: 
      return object(ip::Rotate::getOutputShape<double>(src.bz<double,2>(), a));
    default: PYTHON_ERROR(TypeError, "cannot get shape from unsupporter array of type '%s'", info.str().c_str());
  }
}

template <typename T> static void inner_call1 (ip::Rotate& obj, 
    tp::const_ndarray input, tp::ndarray output) {
  blitz::Array<double,2> output_ = output.bz<double,2>();
  obj(input.bz<T,2>(), output_);
}

static void call1 (ip::Rotate& obj, tp::const_ndarray input,
    tp::ndarray output) {
  const ca::typeinfo& info = input.type();
  switch (info.dtype) {
    case ca::t_uint8: 
      return inner_call1<uint8_t>(obj, input, output);
    case ca::t_uint16:
      return inner_call1<uint16_t>(obj, input, output);
    case ca::t_float64: 
      return inner_call1<double>(obj, input, output);
    default: PYTHON_ERROR(TypeError, "rotation does not support array with type '%s'", info.str().c_str());
  }
}

template <typename T> static void inner_call2 (ip::Rotate& obj, 
    tp::const_ndarray input, tp::ndarray output, double angle) {
  blitz::Array<double,2> output_ = output.bz<double,2>();
  obj(input.bz<T,2>(), output_, angle);
}

static void call2 (ip::Rotate& obj, tp::const_ndarray input,
    tp::ndarray output, double angle) {
  const ca::typeinfo& info = input.type();
  switch (info.dtype) {
    case ca::t_uint8: 
      return inner_call2<uint8_t>(obj, input, output, angle);
    case ca::t_uint16:
      return inner_call2<uint16_t>(obj, input, output, angle);
    case ca::t_float64: 
      return inner_call2<double>(obj, input, output, angle);
    default: PYTHON_ERROR(TypeError, "rotation does not support array with type '%s'", info.str().c_str());
  }
}

template <typename T> static void inner_call3 (ip::Rotate& obj, 
    tp::const_ndarray input, tp::const_ndarray input_mask,
    tp::ndarray output, tp::ndarray output_mask) {
  blitz::Array<double,2> output_ = output.bz<double,2>();
  blitz::Array<bool,2> output_mask_ = output_mask.bz<bool,2>();
  obj(input.bz<T,2>(), input_mask.bz<bool,2>(), output_, output_mask_);
}

static void call3 (ip::Rotate& obj, tp::const_ndarray input,
    tp::const_ndarray input_mask, tp::ndarray output, tp::ndarray output_mask) {
  const ca::typeinfo& info = input.type();
  switch (info.dtype) {
    case ca::t_uint8: 
      return inner_call3<uint8_t>(obj, input, input_mask, output, output_mask);
    case ca::t_uint16:
      return inner_call3<uint16_t>(obj, input, input_mask, output, output_mask);
    case ca::t_float64: 
      return inner_call3<double>(obj, input, input_mask, output, output_mask);
    default: PYTHON_ERROR(TypeError, "rotation (with masks) does not support array with type '%s'", info.str().c_str());
  }
}

template <typename T> static void inner_call4 (ip::Rotate& obj, 
    tp::const_ndarray input, tp::const_ndarray input_mask,
    tp::ndarray output, tp::ndarray output_mask, double angle) {
  blitz::Array<double,2> output_ = output.bz<double,2>();
  blitz::Array<bool,2> output_mask_ = output_mask.bz<bool,2>();
  obj(input.bz<T,2>(), input_mask.bz<bool,2>(), output_, output_mask_, angle);
}

static void call4 (ip::Rotate& obj, tp::const_ndarray input,
    tp::const_ndarray input_mask, tp::ndarray output, tp::ndarray output_mask,
    double angle) {
  const ca::typeinfo& info = input.type();
  switch (info.dtype) {
    case ca::t_uint8: 
      return inner_call4<uint8_t>(obj, input, input_mask, output, output_mask,
          angle);
    case ca::t_uint16:
      return inner_call4<uint16_t>(obj, input, input_mask, output, output_mask,
          angle);
    case ca::t_float64: 
      return inner_call4<double>(obj, input, input_mask, output, output_mask,
          angle);
    default: PYTHON_ERROR(TypeError, "rotation (with masks) does not support array with type '%s'", info.str().c_str());
  }
}

void bind_ip_rotate() {
  enum_<Torch::ip::Rotate::Algorithm>("RotateAlgorithm")
    .value("Shearing", Torch::ip::Rotate::Shearing)
    .value("BilinearInterp", Torch::ip::Rotate::BilinearInterp)
    ;

  class_<ip::Rotate, boost::shared_ptr<ip::Rotate> >("Rotate", rotate_doc, init<const double, optional<const Torch::ip::Rotate::Algorithm> >((arg("rotation_angle"), arg("rotation_algorithm")="Shearing"), "Constructs a Rotate object."))
    .add_property("angle", &ip::Rotate::getAngle, &ip::Rotate::setAngle)
    .add_property("algorithm", &ip::Rotate::getAlgorithm, &ip::Rotate::setAlgorithm)
    .def("__call__", &call1, (arg("self"), arg("input"), arg("output")), "Call an object of this type to perform a rotation of an image.") 
    .def("__call__", &call2, (arg("self"), arg("input"), arg("output"), arg("rotation_angle")), "Call an object of this type to perform a rotation of an image with the given angle.") 
    .def("__call__", &call3, (arg("self"), arg("input"), arg("input_mask"), arg("output"), arg("output_mask")), "Call an object of this type to perform a rotation of an image.") 
    .def("__call__", &call4, (arg("self"), arg("input"), arg("input_mask"), arg("output"), arg("output_mask"), arg("rotation_angle")), "Call an object of this type to perform a rotation of an image.") 


    .def("getOutputShape", &getOutputShape, (arg("input"), arg("rotation_angle")), "Return the required output shape for the given input and rotation angle.")

    .staticmethod("getOutputShape")
    ;

  def("getAngleToHorizontal", (const double (*)(const int, const int, const int, const int))&Torch::ip::getAngleToHorizontal, (arg("left_h"), arg("left_w"), arg("right_h"), arg("right_w")), angle_to_horizontal_doc)
    ;
}
