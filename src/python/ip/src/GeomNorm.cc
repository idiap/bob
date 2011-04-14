/**
 * @file src/python/ip/src/GeomNorm.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 * @date Thu 17 Mar 19:12:40 2011 
 *
 * @brief Binds the GeomNorm class to python
 */

#include <boost/python.hpp>

#include "ip/GeomNorm.h"

using namespace boost::python;
namespace ip = Torch::ip;

static const char* geomnormdoc = "Objects of this class, after configuration, can perform a geometric normalization, for instance, to crop faces from images.";


#define GEOMNORM_CALL_DEF(T) \
  .def("__call__", (void (ip::GeomNorm::*)(const blitz::Array<T,2>&, blitz::Array<double,2>&, const int, const int, const int, const int))&ip::GeomNorm::operator()<T>, (arg("input"), arg("output"), arg("y1"), arg("x1"), arg("y2"), arg("x2")), "Call an object of this type to perform a geometric normalization of an image wrt. the two given points.")

void bind_ip_geomnorm() {
  class_<ip::GeomNorm, boost::shared_ptr<ip::GeomNorm> >("GeomNorm", geomnormdoc, init<const int, const int, const int, const int, const int>((arg("eyes_distance"), arg("height"), arg("width"), arg("border_h"), arg("border_w")), "Constructs a GeomNorm object."))
    GEOMNORM_CALL_DEF(uint8_t)
    GEOMNORM_CALL_DEF(uint16_t)
    GEOMNORM_CALL_DEF(double)
    ;
}
