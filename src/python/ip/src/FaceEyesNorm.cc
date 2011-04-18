/**
 * @file src/python/ip/src/FaceEyesNorm.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 * @date Thu 17 Mar 19:12:40 2011 
 *
 * @brief Binds the FaceEyesNorm class to python
 */

#include <boost/python.hpp>

#include "ip/FaceEyesNorm.h"

using namespace boost::python;
namespace ip = Torch::ip;

static const char* faceeyesnorm_doc = "Objects of this class, after configuration, can extract and normalize faces, given their eye center coordinates.";

#define FACEEYESNORM_CALL_DEF(T) \
  .def("__call__", (void (ip::FaceEyesNorm::*)(const blitz::Array<T,2>&, blitz::Array<double,2>&, const int, const int, const int, const int))&ip::FaceEyesNorm::operator()<T>, (arg("input"), arg("output"), arg("e1_y"), arg("e1_x"), arg("e2_y"), arg("e2_x")), "Call an object of this type to extract a face given the coordinates of the two eye centers.")

void bind_ip_faceeyesnorm() {
  class_<ip::FaceEyesNorm, boost::shared_ptr<ip::FaceEyesNorm> >("FaceEyesNorm", faceeyesnorm_doc, init<const int, const int, const int, const int, const int>((arg("eyes_distance"), arg("crop_height"), arg("crop_width"), arg("crop_eyecenter_offset_h"), arg("crop_eyecenter_offset_w")), "Constructs a FaceEyeNorm object."))
    FACEEYESNORM_CALL_DEF(uint8_t)
    FACEEYESNORM_CALL_DEF(uint16_t)
    FACEEYESNORM_CALL_DEF(double)
    ;
}
