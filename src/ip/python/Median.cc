/**
 * @file ip/python/Median.cc
 * @date Wed Sep 28 20:13:48 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds Median filter into python
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/preprocessor/cat.hpp>
#include "bob/ip/Median.h"

using namespace boost::python;

static const char* medianfilter_doc = "Objects of this class, after configuration, can perform a median filtering operation.";

#define MEDIAN_CLASS(T,N) \
  class_<bob::ip::Median<T> , boost::shared_ptr<bob::ip::Median<T> > >(N, medianfilter_doc, init<const int, const int>((arg("self"), arg("radius_y"), arg("radius_x")), "Constructs a median filter object.")) \
    .def("reset", (void (bob::ip::Median<T>::*)(const int, const int))&bob::ip::Median<T>::reset, (arg("self"), arg("radius_y"), arg("radius_x")), "Updates the kernel dimensions.") \
    .def("__call__", (void (bob::ip::Median<T>::*)(const blitz::Array<T,2>&, blitz::Array<T,2>&))&bob::ip::Median<T>::operator(), (arg("self"), arg("input"), arg("output")), "Call an object of this type to filter an image with a median filter.") \
    .def("__call__", (void (bob::ip::Median<T>::*)(const blitz::Array<T,3>&, blitz::Array<T,3>&))&bob::ip::Median<T>::operator(), (arg("self"), arg("input"), arg("output")), "Call an object of this type to filter an image with a median filter.") \
  ;

void bind_ip_median() {
  MEDIAN_CLASS(uint8_t, "Median_uint8")
  MEDIAN_CLASS(uint16_t, "Median_uint16")
  MEDIAN_CLASS(double, "Median_float64")
}
