/**
 * @file src/python/ip/src/DCTFeatures.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 * @date Thu 17 Mar 19:12:40 2011 
 *
 * @brief Binds the DCT features extractor to python
 */

#include <boost/python.hpp>
#include <vector>
#include "ip/DCTFeatures.h"

using namespace boost::python;
namespace ip = Torch::ip;

static const char* dctdoc = "Objects of this class, after configuration, extract DCT features as described in the paper titled \"Polynomial Features for Robust Face Authentication\", published in 2002.";

template<typename T>
static object dct_apply( ip::DCTFeatures& dct_features, const blitz::Array<T,2>& src) {
  std::vector<blitz::Array<double,1> > dst;
  dct_features( src, dst);
  list t;
  for(std::vector<blitz::Array<double,1> >::const_iterator it=dst.begin(); it!=dst.end(); ++it)
    t.append(*it);
  return t;
}

void bind_ip_dctfeatures() {
  class_<ip::DCTFeatures, boost::shared_ptr<ip::DCTFeatures> >("DCTFeatures", dctdoc, init<const int, const int, const int, const int, const int>((arg("block_h")="8", arg("block_w")="8", arg("overlap_h")="0", arg("overlap_w")="0", arg("n_dct_coefs")="15."), "Constructs a new DCT features extractor."))    
    .def("__call__", &dct_apply<uint8_t>, (arg("self"),arg("input")), "Call an object of this type to extract DCT features.")
    .def("__call__", &dct_apply<uint16_t>, (arg("self"),arg("input")), "Call an object of this type to extract DCT features.")
    .def("__call__", &dct_apply<double>, (arg("self"),arg("input")), "Call an object of this type to extract DCT features.")
    ;
}
