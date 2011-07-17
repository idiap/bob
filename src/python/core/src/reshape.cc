/**
 * @author <a href="mailto:laurent.el-shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Python bindings for the reshape function
 */

#include <boost/python.hpp>
#include "core/python/array_base.h"
#include "core/python/exception.h"

#include "core/reshape.h"

using namespace boost::python;

namespace core = Torch::core;

static const char* RESHAPE2D_DOC_NOCHECK = "Reshapes a 2D array. Does not perform any check on the destination array.";
static const char* RESHAPE2D_DOC_CHECK = "Reshapes a 2D array. Checks are performed on the destination array.";
static const char* RESHAPE2D_DOC_ALLOC = "Reshapes a 2D array (in the matlab way). Allocation of the destination array is performed by this function.";
static const char* RESHAPE_FROM1D_DOC_NOCHECK = "Reshapes a 1D array and generates a 2D array. Does not perform any check on the destination array.";
static const char* RESHAPE_FROM1D_DOC_CHECK = "Reshapes a 1D array and generates a 2D array. Checks are performed on the 2D destination array.";
static const char* RESHAPE_FROM1D_DOC_ALLOC = "Reshapes a 1D array and generates a 2D array (in the matlab way). Allocation of the destination array is performed by this function.";
static const char* RESHAPE_TO1D_DOC_NOCHECK = "Reshapes a 2D array and generates a 2D array. Does not perform any check on the destination array.";
static const char* RESHAPE_TO1D_DOC_CHECK = "Reshapes a 2D array and generates a 2D array. Checks are performed on the 1D destination array.";
static const char* RESHAPE_TO1D_DOC_ALLOC = "Reshapes a 2D array and generates a 2D array (in the matlab way). Allocation of the destination array is performed by this function.";

template<typename T>
static blitz::Array<T,2> reshape2d_r(const blitz::Array<T,2>& A, const int m, const int n) {
  blitz::Array<T,2> B(m, n);
  core::reshape(A, B);
  return B;
}

template<typename T>
static blitz::Array<T,2> reshape_from1d_r(const blitz::Array<T,1>& A, const int m, const int n) {
  blitz::Array<T,2> B;
  B.resize(m, n);
  core::reshape(A, B);
  return B;
}

template<typename T>
static blitz::Array<T,1> reshape_to1d_r(const blitz::Array<T,2>& A) {
  blitz::Array<T,1> B(A.extent(0)*A.extent(1));
  core::reshape(A, B);
  return B;
}

/**
 * This template method simplifies the declaration of python bindings.
 */
template<typename T> void def_reshape() {
  def("reshape_", (void (*)(const blitz::Array<T,2>&, blitz::Array<T,2>&))&core::reshape_, (arg("A"), arg("B")), RESHAPE2D_DOC_NOCHECK);
  def("reshape", (void (*)(const blitz::Array<T,2>&, blitz::Array<T,2>&))&core::reshape, (arg("A"), arg("B")), RESHAPE2D_DOC_CHECK);
  def("reshape", &reshape2d_r<T>, (arg("A"), arg("m"), arg("n")), RESHAPE2D_DOC_ALLOC);

  def("reshape_", (void (*)(const blitz::Array<T,1>&, blitz::Array<T,2>&))&core::reshape_, (arg("a"), arg("B")), RESHAPE_FROM1D_DOC_NOCHECK);
  def("reshape", (void (*)(const blitz::Array<T,1>&, blitz::Array<T,2>&))&core::reshape, (arg("a"), arg("B")), RESHAPE_FROM1D_DOC_CHECK);
  def("reshape", &reshape_from1d_r<T>, (arg("a"), arg("m"), arg("n")), RESHAPE_FROM1D_DOC_ALLOC);

  def("reshape_", (void (*)(const blitz::Array<T,2>&, blitz::Array<T,1>&))&core::reshape_, (arg("A"), arg("B")), RESHAPE_TO1D_DOC_NOCHECK);
  def("reshape", (void (*)(const blitz::Array<T,2>&, blitz::Array<T,1>&))&core::reshape, (arg("A"), arg("B")), RESHAPE_TO1D_DOC_CHECK);
  def("reshape", &reshape_to1d_r<T>, (arg("A")), RESHAPE_TO1D_DOC_ALLOC);
}

void bind_core_reshape() {
  Torch::core::python::CxxToPythonTranslatorPar2<Torch::core::ReshapeDifferentNumberOfElements, Torch::core::Exception, const int, const int>("ReshapeDifferentNumberOfElements", "This exception is thrown when the source array and the destination array which are passed to the reshape function do not have the same number of elements.");

  def_reshape<bool>();
  def_reshape<int8_t>();
  def_reshape<int16_t>();
  def_reshape<int32_t>();
  def_reshape<int64_t>();
  def_reshape<uint8_t>();
  def_reshape<uint16_t>();
  def_reshape<uint32_t>();
  def_reshape<uint64_t>();
  def_reshape<float>();
  def_reshape<double>();
  def_reshape<long double>();
  def_reshape<std::complex<float> >();
  def_reshape<std::complex<double> >();
  def_reshape<std::complex<long double> >();
}
