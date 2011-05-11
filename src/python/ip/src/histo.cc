/**
 * @file python/ip/src/histo.cc 
 * @author <a href="mailto:Francois.Moulin@idiap.ch">Francois Moulin</a> 
 *
 * @brief Binds histogram to python 
 */

#include <boost/python.hpp>

#include "ip/histo.h"
#include "core/python/exception.h"

using namespace boost::python;
namespace ip = Torch::ip;
//namespace tpy = Torch::core::python;

template<typename T>
boost::shared_ptr<blitz::Array<uint64_t, 1> > histogram1(blitz::Array<T, 2>& input) {
  int size = Torch::ip::detail::getHistoSize<T>();
  boost::shared_ptr<blitz::Array<uint64_t, 1> > output = boost::shared_ptr<blitz::Array<uint64_t, 1> >(new blitz::Array<uint64_t, 1>(size));
  
  ip::histogram(input, *output.get());
  return output;
}


template<typename T>
void histogram2(blitz::Array<T, 2>& input, blitz::Array<uint64_t, 1> output) {
  ip::histogram(input, output, false);
}

template<typename T>
void histogram2Acc(blitz::Array<T, 2>& input, blitz::Array<uint64_t, 1> output) {
  ip::histogram(input, output, true);
}

#define histo_uint8_uint16(name, fct, ...)\
def(name, fct<uint8_t>, __VA_ARGS__);\
def(name, fct<uint16_t>, __VA_ARGS__)

#define histo_all_types_acc(name, fct, acc, ...)\
def(name, fct<uint8_t, acc>, __VA_ARGS__);\
def(name, fct<uint16_t, acc>, __VA_ARGS__);\
def(name, fct<uint32_t, acc>, __VA_ARGS__);\
def(name, fct<uint64_t, acc>, __VA_ARGS__);\
def(name, fct<int8_t, acc>, __VA_ARGS__);\
def(name, fct<int16_t, acc>, __VA_ARGS__);\
def(name, fct<int32_t, acc>, __VA_ARGS__);\
def(name, fct<int64_t, acc>, __VA_ARGS__);\
def(name, fct<float, acc>, __VA_ARGS__);\
def(name, fct<double, acc>, __VA_ARGS__);\
def(name, fct<long double, acc>, __VA_ARGS__)


#define histo_all_types(name, fct, ...)\
histo_all_types_acc(name, fct, false, __VA_ARGS__);\
histo_all_types_acc(name "Accumulate", fct, true, __VA_ARGS__)

template<typename T, bool ACCUMULATE>
void histogram3(blitz::Array<T, 2>& src, blitz::Array<uint64_t, 1>& histo, T max) {
  ip::histogram<T>(src, histo, 0, max, (int)max + 1, ACCUMULATE);
}

template<typename T, bool ACCUMULATE>
void histogram4(blitz::Array<T, 2>& src, blitz::Array<uint64_t, 1>& histo, T min, T max) {
  ip::histogram<T>(src, histo, min, max, (int)(max-min) + 1, ACCUMULATE);
}

template<typename T, bool ACCUMULATE>
void histogram5(blitz::Array<T, 2>& src, blitz::Array<uint64_t, 1>& histo, T min, T max, uint32_t nb_bins) {
  ip::histogram<T>(src, histo, min, max, nb_bins, ACCUMULATE);
}



template<typename T, bool ACCUMULATE>
boost::shared_ptr<blitz::Array<uint64_t, 1> > histogram3_(blitz::Array<T, 2>& src, T max) {
  int size = (int)max + 1;
  if (size < 0) {
    throw ip::UnsupportedTypeForHistogram(Torch::core::array::getElementType<T>());
  }
  
  boost::shared_ptr<blitz::Array<uint64_t, 1> > output = boost::shared_ptr<blitz::Array<uint64_t, 1> >(new blitz::Array<uint64_t, 1>(size));
  ip::histogram<T>(src, *output.get(), 0, max, size, false);

  return output;
}


template<typename T, bool ACCUMULATE>
boost::shared_ptr<blitz::Array<uint64_t, 1> > histogram4_(blitz::Array<T, 2>& src, T min, T max) {
  int size = (int)(max - min) + 1;
  if (size < 0) {
    throw ip::UnsupportedTypeForHistogram(Torch::core::array::getElementType<T>());
  }
  
  boost::shared_ptr<blitz::Array<uint64_t, 1> > output = boost::shared_ptr<blitz::Array<uint64_t, 1> >(new blitz::Array<uint64_t, 1>(size));
  ip::histogram<T>(src, *output.get(), min, max, size, false);

  return output;
}


template<typename T, bool ACCUMULATE>
boost::shared_ptr<blitz::Array<uint64_t, 1> > histogram5_(blitz::Array<T, 2>& src, T min, T max, uint32_t nb_bins) {
  int size = nb_bins;
  if (size < 0) {
    throw ip::UnsupportedTypeForHistogram(Torch::core::array::getElementType<T>());
  }
  
  boost::shared_ptr<blitz::Array<uint64_t, 1> > output = boost::shared_ptr<blitz::Array<uint64_t, 1> >(new blitz::Array<uint64_t, 1>(size));
  ip::histogram<T>(src, *output.get(), min, max, nb_bins, false);

  return output;
}


void bind_ip_histogram()
{
  //Exceptions for this functionality
  Torch::core::python::CxxToPythonTranslatorPar<Torch::ip::UnsupportedTypeForHistogram, Torch::core::Exception , Torch::core::array::ElementType>("UnsupportedTypeForHistogram", "This exception is thrown when the histogram computation for a particular type is not implemented in torch");
  Torch::core::python::CxxToPythonTranslator<Torch::ip::InvalidArgument, Torch::core::Exception>("InvalidArgument", "This exception is thrown when a function argument is invalid");

  histo_uint8_uint16("histogram", &histogram1, args("input"), "Compute an histogram of a 2D array");
  
  histo_uint8_uint16("histogram", &histogram2,
                     (arg("src"), arg("histo")),
                     "Compute an histogram of a 2D array.\n"
                     "histo must have a size of 2^N-1 elements, where N is the number of bits in input.");
  
  histo_uint8_uint16("histogramAccumulate", &histogram2Acc,
                     (arg("src"), arg("histo")),
                     "Compute an histogram of a 2D array and add it to histo.\n"
                     "histo must have a size of 2^N-1 elements, where N is the number of bits in input.");

  histo_all_types("histogram", &histogram3,
                  (arg("src"), arg("histo"), arg("max")),
                  "Compute an histogram of a 2D array.\n"
                  "src elements are in range [0, max] (max >= 0)\n"
                  "histo must have a size of max elements");
  
  histo_all_types("histogram", &histogram4,
                  (arg("src"), arg("histo"), arg("min"), arg("max")),
                  "Compute an histogram of a 2D array.\n"
                  "src elements are in range [min, max] (max >= min)\n"
                  "histo must have a size of max-min elements");
  
  histo_all_types("histogram", &histogram5,
                  (arg("src"), arg("histo"), arg("min"), arg("max"), arg("nb_bins")),
                  "Compute an histogram of a 2D array.\n"
                  "src elements are in range [min, max] (max >= min)\n"
                  "histo must have a size of nb_bins elements");

  
  histo_all_types_acc("histogram", &histogram3_, false,
                      (arg("src"), arg("max")),
                      "Return an histogram of a 2D array.\n"
                      "src elements are in range [0, max] (max >= 0)\n");
  
  histo_all_types_acc("histogram", &histogram4_, false,
                      (arg("src"), arg("min"), arg("max")),
                      "Return an histogram of a 2D array.\n"
                      "src elements are in range [min, max] (max >= min)\n");
  
  histo_all_types_acc("histogram", &histogram5_, false,
                      (arg("src"), arg("min"), arg("max"), arg("nb_bins")),
                      "Return an histogram of a 2D array.\n"
                      "src elements are in range [min, max] (max >= min)\n");

}
