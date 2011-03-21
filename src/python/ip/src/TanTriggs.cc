/**
 * @file src/python/ip/src/TanTriggs.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 * @date Thu 17 Mar 19:12:40 2011 
 *
 * @brief Binds the Tan and Triggs preprocessing filter to python
 */

#include <boost/python.hpp>

#include "ip/TanTriggs.h"

using namespace boost::python;
namespace ip = Torch::ip;

static const char* ttdoc = "Objects of this class, after configuration, can preprocess images. It does this using the method described by Tan and Triggs in the paper titled \" Enhanced_Local_Texture_Feature_Sets for_Face_Recognition_Under_Difficult_Lighting_Conditions\", published in 2007";

#define TANTRIGGS_CALL_DEF(T) \
  .def("__call__", (void (ip::TanTriggs::*)(const blitz::Array<T,2>&, blitz::Array<double,2>&))&ip::TanTriggs::operator()<T>, (arg("input"), arg("output")), "Call an object of this type to compute a preprocessed image.")

void bind_ip_tantriggs() {
  class_<ip::TanTriggs, boost::shared_ptr<ip::TanTriggs> >("TanTriggs", ttdoc, init<optional<const double, const double, const double, const double, const double> >((arg("gamma")="0.1", arg("sigma0")="1", arg("sigma1")="2", arg("size")="2", arg("threshold")="10.", arg("alpha")="0.1"), "Constructs a new Tan and Triggs filter."))
    //.add_property("test", &ip::TanTriggs::getTest, &ip::TanTriggs::setTest)
    TANTRIGGS_CALL_DEF(uint8_t)
    TANTRIGGS_CALL_DEF(uint16_t)
    TANTRIGGS_CALL_DEF(double)
    ;
}
