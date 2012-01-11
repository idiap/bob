/**
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @date Wed 28 Sep 2011
 *
 * @brief Binds Median filter into python
 */

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/preprocessor/cat.hpp>
#include "ip/Median.h"

using namespace boost::python;
namespace ip = bob::ip;


static const char* medianfilter_doc = "Objects of this class, after configuration, can perform a median filtering operation.";

#define MEDIAN_CLASS(T,N) \
  class_<ip::Median<T> , boost::shared_ptr<ip::Median<T> > >(N, medianfilter_doc, init<const int, const int>((arg("radius_y"), arg("radius_x")), "Constructs a median filter object.")) \
    .def("reset", (void (ip::Median<T>::*)(const int, const int))&ip::Median<T>::reset, (arg("radius_y"), arg("radius_x")), "Updates the kernel dimensions.") \
    .def("__call__", (void (ip::Median<T>::*)(const blitz::Array<T,2>&, blitz::Array<T,2>&))&ip::Median<T>::operator(), (arg("input"), arg("output")), "Call an object of this type to filter an image with a median filter.") \
    .def("__call__", (void (ip::Median<T>::*)(const blitz::Array<T,3>&, blitz::Array<T,3>&))&ip::Median<T>::operator(), (arg("input"), arg("output")), "Call an object of this type to filter an image with a median filter.") \
  ;

void bind_ip_median() {
  MEDIAN_CLASS(uint8_t, "Median_uint8")
  MEDIAN_CLASS(uint16_t, "Median_uint16")
  MEDIAN_CLASS(double, "Median_float64")
}
