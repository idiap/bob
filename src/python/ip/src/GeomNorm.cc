/**
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @date Thu 17 Mar 19:12:40 2011 
 *
 * @brief Binds the GeomNorm class to python
 */

#include "core/python/ndarray.h"
#include "ip/GeomNorm.h"
#include "ip/maxRectInMask.h"

using namespace boost::python;
namespace ip = Torch::ip;
namespace tp = Torch::python;
namespace ca = Torch::core::array;

static const char* GEOMNORM_DOC = "Objects of this class, after configuration, can perform a geometric normalization.";
static const char* MAXRECTINMASK2D_DOC = "Given a 2D mask (a 2D blitz array of booleans), compute the maximum rectangle which only contains true values.";

template <typename T> static void inner_call1 (ip::GeomNorm& obj, 
    tp::const_ndarray input, tp::ndarray output,
    int a, int b, int c, int d) {
  blitz::Array<double,2> output_ = output.bz<double,2>();
  obj(input.bz<T,2>(), output_, a,b,c,d);
}

static void call1 (ip::GeomNorm& obj, tp::const_ndarray input,
    tp::ndarray output, int a, int b, int c, int d) {
  const ca::typeinfo& info = input.type();
  switch (info.dtype) {
    case ca::t_uint8: 
      return inner_call1<uint8_t>(obj, input, output, a,b,c,d);
    case ca::t_uint16:
      return inner_call1<uint16_t>(obj, input, output, a,b,c,d);
    case ca::t_float64: 
      return inner_call1<double>(obj, input, output, a,b,c,d);
    default: PYTHON_ERROR(TypeError, "geometric normalization does not support array with type '%s'", info.str().c_str());
  }
}

template <typename T> static void inner_call2 (ip::GeomNorm& obj, 
    tp::const_ndarray input, tp::const_ndarray input_mask,
    tp::ndarray output, tp::ndarray output_mask,
    int a, int b, int c, int d) {
  blitz::Array<double,2> output_ = output.bz<double,2>();
  blitz::Array<bool,2> output_mask_ = output_mask.bz<bool,2>();
  obj(input.bz<T,2>(), input_mask.bz<bool,2>(), output_, output_mask_,
      a, b, c, d);
}

static void call2 (ip::GeomNorm& obj, tp::const_ndarray input,
    tp::const_ndarray input_mask, tp::ndarray output, tp::ndarray output_mask,
    int a, int b, int c, int d) {
  const ca::typeinfo& info = input.type();
  switch (info.dtype) {
    case ca::t_uint8: 
      return inner_call2<uint8_t>(obj, input, input_mask, output, output_mask, a, b, c, d);
    case ca::t_uint16:
      return inner_call2<uint16_t>(obj, input, input_mask, output, output_mask, a, b, c, d);
    case ca::t_float64: 
      return inner_call2<double>(obj, input, input_mask, output, output_mask, a, b, c, d);
    default: PYTHON_ERROR(TypeError, "geometric normalization (with masks) does not support array with type '%s'", info.str().c_str());
  }
}

void bind_ip_geomnorm() {
  class_<ip::GeomNorm, boost::shared_ptr<ip::GeomNorm> >("GeomNorm", GEOMNORM_DOC, init<const double, const double, const int, const int, const int, const int>((arg("rotation_angle"), arg("scaling_factor"), arg("crop_height"), arg("crop_width"), arg("crop_offset_h"), arg("crop_offset_w")), "Constructs a GeomNorm object."))
    .add_property("rotation_angle", &ip::GeomNorm::getRotationAngle, &ip::GeomNorm::setRotationAngle)
    .add_property("scaling_factor", &ip::GeomNorm::getScalingFactor, &ip::GeomNorm::setScalingFactor)
    .add_property("crop_height", &ip::GeomNorm::getCropHeight, &ip::GeomNorm::setCropHeight)
    .add_property("crop_width", &ip::GeomNorm::getCropWidth, &ip::GeomNorm::setCropWidth)
    .add_property("crop_offset_h", &ip::GeomNorm::getCropOffsetH, &ip::GeomNorm::setCropOffsetH)
    .add_property("crop_offset_w", &ip::GeomNorm::getCropOffsetW, &ip::GeomNorm::setCropOffsetW)
  .def("__call__", &call1, (arg("input"), arg("output"), arg("rotation_center_y"), arg("rotation_center_x"), arg("crop_ref_y"), arg("crop_ref_x")), "Call an object of this type to perform a geometric normalization of an image wrt. the two given points.")
  .def("__call__", &call2, (arg("input"), arg("input_mask"), arg("output"), arg("output_mask"), arg("rotation_center_y"), arg("rotation_center_x"), arg("crop_ref_y"), arg("crop_ref_x")), "Call an object of this type to perform a geometric normalization of an image wrt. the two given points, taking mask into account.")
    ;

  def("maxRectInMask", (const blitz::TinyVector<int,4> (*)(const blitz::Array<bool,2>&))&Torch::ip::maxRectInMask, (("src")), MAXRECTINMASK2D_DOC); 
}
