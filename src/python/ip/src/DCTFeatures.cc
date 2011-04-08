/**
 * @file src/python/ip/src/DCTFeatures.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 * @date Thu 17 Mar 19:12:40 2011 
 *
 * @brief Binds the DCT features extractor to python
 */

#include <boost/python.hpp>

#include "ip/DCTFeatures.h"

using namespace boost::python;
namespace ip = Torch::ip;

static const char* dctdoc = "Objects of this class, after configuration, extract DCT features as described in the paper titled \"Polynomial Features for Robust Face Authentication\", published in 2002.";

#define S_DCTFEATURES_CALL_DEF(T) \
  static list dct_apply( ip::DCTFeatures& dct_features, const blitz::Array<T,2>& src) { \
    std::list<blitz::Array<double,1> > dst; \
    dct_features( src, dst); \
    list t; \
    for(std::list<blitz::Array<double,1> >::const_iterator it=dst.begin(); it!=dst.end(); ++it) \
      t.append(*it); \
    return t; \
  }

S_DCTFEATURES_CALL_DEF(uint8_t)
S_DCTFEATURES_CALL_DEF(uint16_t)
S_DCTFEATURES_CALL_DEF(double)


#define DCTFEATURES_CALL_DEF(T) \
  .def("__call__", (list (*)(ip::DCTFeatures&, const blitz::Array<T,2>&))&dct_apply, (arg("self"),arg("input")), "Call an object of this type to extract DCT features.")
/* TODO: cleanup
#define DCTFEATURES_CALL_DEF_old(T) \
  .def("__call__", (void (ip::DCTFeatures::*)(const blitz::Array<T,2>&, std::list<blitz::Array<double,1> >&))&ip::DCTFeatures::operator()<T>, (arg("input"), arg("output")), "Call an object of this type to extract DCT features.")
*/

void bind_ip_dctfeatures() {
  class_<ip::DCTFeatures, boost::shared_ptr<ip::DCTFeatures> >("DCTFeatures", dctdoc, init<const int, const int, const int, const int, const int>((arg("block_h")="8", arg("block_w")="8", arg("overlap_h")="0", arg("overlap_w")="0", arg("n_dct_coefs")="15."), "Constructs a new DCT features extractor."))    
// TODO: cleanup .def("__call__", (list (*)(ip::DCTFeatures&, const blitz::Array<double,2>&))&dct_apply, (arg("self"),arg("input")), "Call an object of this type to extract DCT features.")
    DCTFEATURES_CALL_DEF(uint8_t)
    DCTFEATURES_CALL_DEF(uint16_t)
    DCTFEATURES_CALL_DEF(double)
    ;
}
