/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Support for vectors of scalars 
 */

#include <complex>
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

using namespace boost::python;

#define BINDER(T,E) class_<std::vector<T> >(BOOST_PP_STRINGIZE(E)).def(vector_indexing_suite<std::vector<T> >())

void bind_core_vectors () {
  BINDER(std::string, string);
  BINDER(bool, bool);
  BINDER(int8_t, int8);
  BINDER(int16_t, int16);
  BINDER(int32_t, int32);
  BINDER(int64_t, int64);
  BINDER(uint8_t, uint8);
  BINDER(uint16_t, uint16);
  BINDER(uint32_t, uint32);
  BINDER(uint64_t, uint64);
  BINDER(float, float32);
  BINDER(double, float64);
  BINDER(long double, float128);
  BINDER(std::complex<float>, complex64);
  BINDER(std::complex<double>, complex128);
  BINDER(std::complex<long double>, complex256);
}
