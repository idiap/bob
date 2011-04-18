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


void bind_ip_histogram()
{
  //Exceptions for this functionality
  Torch::core::python::CxxToPythonTranslatorPar<Torch::ip::UnsupportedTypeForHistogram, Torch::core::Exception , Torch::core::array::ElementType>("UnsupportedTypeForHistogram", "This exception is thrown when the histogram computation for a particular type is not implemented in torch");
  Torch::core::python::CxxToPythonTranslator<Torch::ip::InvalidArgument, Torch::core::Exception>("InvalidArgument", "This exception is thrown when a function argument is invalid");
  
  def("histogram", (void (*)(blitz::Array<uint8_t, 2>&, blitz::Array<uint32_t, 1>&))&ip::histogram<uint8_t>, (arg("src"), arg("histo")), "Compute an histogram of a 2D array.\nhisto must have a size of 256 elements.");
  def("histogram", (void (*)(blitz::Array<uint16_t, 2>&, blitz::Array<uint32_t, 1>&))&ip::histogram<uint16_t>, (arg("src"), arg("histo")), "Compute an histogram of a 2D array.\nhisto must have a size of 65536 elements.");
  
  static const char* desc = "Compute an histogram of a 2D array.\n"
                            "min and max define the range of src values.\n"
                            "histo must have nb_bins elements and min < max.";
                            
  //blitz::Array<T, 2>& src, blitz::Array<uint32_t, 1>& histo, T min, T max, uint32_t nb_bins
  def("histogram", (void (*)(blitz::Array<uint8_t, 2>&, blitz::Array<uint32_t, 1>&, uint8_t, uint8_t, uint32_t))&ip::histogram<uint8_t>, (arg("src"), arg("histo"), arg("min"), arg("max"), arg("nb_bins")), desc);
  def("histogram", (void (*)(blitz::Array<uint16_t, 2>&, blitz::Array<uint32_t, 1>&, uint16_t, uint16_t, uint32_t))&ip::histogram<uint16_t>, (arg("src"), arg("histo"), arg("min"), arg("max"), arg("nb_bins")), desc);
  def("histogram", (void (*)(blitz::Array<uint32_t, 2>&, blitz::Array<uint32_t, 1>&, uint32_t, uint32_t, uint32_t))&ip::histogram<uint32_t>, (arg("src"), arg("histo"), arg("min"), arg("max"), arg("nb_bins")), desc);
  def("histogram", (void (*)(blitz::Array<uint64_t, 2>&, blitz::Array<uint32_t, 1>&, uint64_t, uint64_t, uint32_t))&ip::histogram<uint64_t>, (arg("src"), arg("histo"), arg("min"), arg("max"), arg("nb_bins")), desc);
  def("histogram", (void (*)(blitz::Array<int8_t, 2>&, blitz::Array<uint32_t, 1>&, int8_t, int8_t, uint32_t))&ip::histogram<int8_t>, (arg("src"), arg("histo"), arg("min"), arg("max"), arg("nb_bins")), desc);
  def("histogram", (void (*)(blitz::Array<int16_t, 2>&, blitz::Array<uint32_t, 1>&, int16_t, int16_t, uint32_t))&ip::histogram<int16_t>, (arg("src"), arg("histo"), arg("min"), arg("max"), arg("nb_bins")), desc);
  def("histogram", (void (*)(blitz::Array<int32_t, 2>&, blitz::Array<uint32_t, 1>&, int32_t, int32_t, uint32_t))&ip::histogram<int32_t>, (arg("src"), arg("histo"), arg("min"), arg("max"), arg("nb_bins")), desc);
  def("histogram", (void (*)(blitz::Array<int64_t, 2>&, blitz::Array<uint32_t, 1>&, int64_t, int64_t, uint32_t))&ip::histogram<int64_t>, (arg("src"), arg("histo"), arg("min"), arg("max"), arg("nb_bins")), desc);
  def("histogram", (void (*)(blitz::Array<float, 2>&, blitz::Array<uint32_t, 1>&, float, float, uint32_t))&ip::histogram<float>, (arg("src"), arg("histo"), arg("min"), arg("max"), arg("nb_bins")), desc);
  def("histogram", (void (*)(blitz::Array<double, 2>&, blitz::Array<uint32_t, 1>&, double, double, uint32_t))&ip::histogram<double>, (arg("src"), arg("histo"), arg("min"), arg("max"), arg("nb_bins")), desc);
  def("histogram", (void (*)(blitz::Array<long double, 2>&, blitz::Array<uint32_t, 1>&, long double, long double, uint32_t))&ip::histogram<long double>, (arg("src"), arg("histo"), arg("min"), arg("max"), arg("nb_bins")), desc);
  
}
