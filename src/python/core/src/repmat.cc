/**
 * @author <a href="mailto:laurent.el-shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Python bindings for torch::core::cast
 */

#include <boost/python.hpp>
#include "core/python/array_base.h"

#include "core/repmat.h"

using namespace boost::python;

namespace core = Torch::core;

static const char* REPMAT2D_DOC_NOCHECK = "Replicates a 2D array. Does not perform any check on the destination array.";
static const char* REPMAT2D_DOC_CHECK = "Replicates a 2D array. Checks are performed on the destination array.";
static const char* REPMAT2D_DOC_ALLOC = "Replicates a 2D array (in the matlab way). Allocation of the destination array is performed by this function.";
static const char* REPMAT1D_DOC_NOCHECK = "Replicates a 1D array and generates a 2D array. Does not perform any check on the destination array.";
static const char* REPMAT1D_DOC_CHECK = "Replicates a 1D array and generates a 2D array. Checks are performed on the 2D destination array.";
static const char* REPMAT1D_DOC_ALLOC = "Replicates a 1D array and generates a 2D array (in the matlab way). Allocation of the destination array is performed by this function.";
static const char* REPVEC_DOC_NOCHECK = "Replicates a 1D array and generates a larger 1D array. Does not perform any check on the destination array.";
static const char* REPVEC_DOC_CHECK = "Replicates a 1D array and generates a larger 1D array. Checks are performed on the 2D destination array.";
static const char* REPVEC_DOC_ALLOC = "Replicates a 1D array and generates a larger 1D array (in the matlab way). Allocation of the destination array is performed by this function.";

template<typename T>
static blitz::Array<T,2> repmat2d_r(const blitz::Array<T,2>& A, const int m, const int n) {
  blitz::Array<T,2> B(m*A.extent(0), n*A.extent(1));
  core::repmat_(A, B);
  return B;
}

template<typename T>
static blitz::Array<T,2> repmat1d_r(const blitz::Array<T,1>& A, const int m, const int n, bool row_vector_src) {
  blitz::Array<T,2> B;
  if(row_vector_src)
    B.resize(m, n*A.extent(0));
  else
    B.resize(m*A.extent(0), n);
  core::repmat_(A, B, row_vector_src);
  return B;
}

template<typename T>
static blitz::Array<T,1> repvec_r(const blitz::Array<T,1>& A, const int m) {
  blitz::Array<T,1> B(m*A.extent(0));
  core::repvec_(A, B);
  return B;
}

/**
 * This template method simplifies the declaration of python bindings.
 */
template<typename T> void def_repmat() {
  def("repmat_", (void (*)(const blitz::Array<T,2>&, blitz::Array<T,2>&))&core::repmat_, (arg("A"), arg("B")), REPMAT2D_DOC_NOCHECK);
  def("repmat", (void (*)(const blitz::Array<T,2>&, blitz::Array<T,2>&))&core::repmat, (arg("A"), arg("B")), REPMAT2D_DOC_CHECK);
  def("repmat", &repmat2d_r<T>, (arg("A"), arg("m"), arg("n")), REPMAT2D_DOC_ALLOC);

  def("repmat_", (void (*)(const blitz::Array<T,1>&, blitz::Array<T,2>&, const bool))&core::repmat_, (arg("a"), arg("B"), arg("row_vector_src")=false), REPMAT1D_DOC_NOCHECK);
  def("repmat", (void (*)(const blitz::Array<T,1>&, blitz::Array<T,2>&, const bool))&core::repmat, (arg("a"), arg("B"), arg("row_vector_src")=false), REPMAT1D_DOC_CHECK);
  def("repmat", &repmat1d_r<T>, (arg("a"), arg("m"), arg("n"), arg("row_vector_src")=false), REPMAT1D_DOC_ALLOC);

  def("repvec_", (void (*)(const blitz::Array<T,1>&, blitz::Array<T,1>&))&core::repvec_, (arg("A"), arg("B")), REPVEC_DOC_NOCHECK);
  def("repvec", (void (*)(const blitz::Array<T,1>&, blitz::Array<T,1>&))&core::repvec_, (arg("A"), arg("B")), REPVEC_DOC_CHECK);
  def("repvec", &repvec_r<T>, (arg("A"), arg("m")), REPVEC_DOC_ALLOC);
}

void bind_core_repmat() {
  def_repmat<bool>();
  def_repmat<int8_t>();
  def_repmat<int16_t>();
  def_repmat<int32_t>();
  def_repmat<int64_t>();
  def_repmat<uint8_t>();
  def_repmat<uint16_t>();
  def_repmat<uint32_t>();
  def_repmat<uint64_t>();
  def_repmat<float>();
  def_repmat<double>();
  def_repmat<long double>();
  def_repmat<std::complex<float> >();
  def_repmat<std::complex<double> >();
  def_repmat<std::complex<long double> >();
}
