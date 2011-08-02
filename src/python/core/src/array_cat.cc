/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Fri 29 Jul 08:55:21 2011
 *
 * @brief Python bindings for array concatenation
 */

#include <boost/python.hpp>
#include "core/array_type.h"
#include "core/array_cat.h"

using namespace boost::python;
namespace array = Torch::core::array;

static const char DCOPY_DOC[] = "Copies the data from the source array into the destination array, along a certain dimension and starting from a certain position. This method is not a generic copying facility. It helps concatenation. Both arrays have to have the same shape except perhaps along the copying dimension.";

static const char DCOPY_DOC_[] = "Copies the data from the source array into the destination array, along a certain dimension and starting from a certain position. This method is not a generic copying facility. It helps concatenation. Both arrays have to have the same shape except perhaps along the copying dimension. **This version does not run any checks on user input.**";

static const char DCOPY_DOC1[] = "Copies the data from the source array into the destination array, along a certain dimension and starting from a certain position. This method is not a generic copying facility. It helps concatenation. The source arrays have fit precisely on the destination array except, perhaps along the copying dimension.";

static const char DCOPY_DOC1_[] = "Copies the data from the source array into the destination array, along a certain dimension and starting from a certain position. This method is not a generic copying facility. It helps concatenation. The source arrays have fit precisely on the destination array except, perhaps along the copying dimension. **This version doe snot run any checks on user input.**";

static const char CAT_DOC[] = "Concatenates arrays with the same number of dimensions along a particular dimension. All input arrays must have the same shape except along the concatenation dimension. The output array will have the same number of dimensions of the input arrays.";

static const char CAT_DOC_[] = "Concatenates arrays with the same number of dimensions along a particular dimension. All input arrays must have the same shape except along the concatenation dimension. The output array will have the same number of dimensions of the input arrays. **This version does not run any checks on user input.**";

static const char CAT_DOC_PY[] = "Concatenates arrays with the same number of dimensions along a particular dimension. All input arrays must have the same shape except along the concatenation dimension. The output array is allocated dynamically and will have the same number of dimensions of the input arrays..";

static const char STACK_DOC[] = "Stacks arrays with the same number of dimensions along the first dimension of the destination array. All input arrays must have the same shape. In this variant the input arrays are concatenated to create an array with an extra dimension. Each entry in this new array contains the input arrays organized in the same order they were passed. The output array will have an additional dimension compared to the number of dimensions of the input arrays.";

static const char STACK_DOC_[] = "Concatenates arrays with the same number of dimensions along the first dimension of the destination array. All input arrays must have the same shape. In this variant the input arrays are concatenated to create an array with an extra dimension. Each entry in this new array contains the input arrays organized in the same order they were passed. The output array will have an additional dimension compared to the number of dimensions of the input arrays. **This version does not run any checks on user input.**";

static const char STACK_DOC_PY[] = "Concatenates arrays with the same number of dimensions along the first dimension of the (returned) destionation array. All input arrays must have the same shape. The output array is allocated dynamically and will be composed of N+1 dimensions (if N is the number of input dimensions for the input arrays). The input arrays will be concatenated along the given dimension.";

template <typename T, int N> static blitz::Array<T,N>
cat(const std::vector<blitz::Array<T,N> >& source, int D) {
  if (!source.size()) return blitz::Array<T,N>();
  blitz::TinyVector<int,N> shape = source[0].shape();
  shape(D) = 0;
  for (size_t i=0; i<source.size(); ++i) shape(D) += source[i].extent(D);
  blitz::Array<T,N> dest(shape);
  array::cat(source, dest, D);
  return dest;
}

template <typename T, int N> static blitz::Array<T,N> 
stack(const std::vector<blitz::Array<T,N-1> >& source) {
  if (!source.size()) return blitz::Array<T,N>();
  blitz::TinyVector<int,N> shape;
  for (int k=1; k<N; ++k) shape(k) = source[0].shape()(k-1);
  shape(0) = source.size();
  blitz::Array<T,N> dest(shape);
  array::stack(source, dest);
  return dest;
}

template <typename T> static void bind_cat1() {
  typedef blitz::Array<T,1> aN;
  typedef std::vector<aN> aNv;

  def("dcopy_", (void (*)(const aN&,  aN&, int, int))&array::dcopy_<T,1>,
      (arg("source"), arg("dest"), arg("dim"), arg("pos")), DCOPY_DOC_);
  def("dcopy" , (void (*)(const aN&,  aN&, int, int))&array::dcopy<T,1>,
      (arg("source"), arg("dest"), arg("dim"), arg("pos")), DCOPY_DOC);

  def("cat_"  , (void (*)(const aNv&, aN&, int))&array::cat_<T,1>,
      (arg("source"), arg("dest"), arg("dim")), CAT_DOC_);
  def("cat"   , (void (*)(const aNv&, aN&, int))&array::cat<T,1>,
      (arg("source"), arg("dest"), arg("dim")), CAT_DOC_);

  def("cat"   , (aN   (*)(const aNv&, int))&cat<T,1>,
      (arg("source"), arg("dim")), CAT_DOC_PY);

}

template <typename T, int N> static void bind_cat() {
  typedef blitz::Array<T,N> aN;
  typedef std::vector<aN> aNv;
  typedef blitz::Array<T,N-1> aN1;
  typedef std::vector<aN1> aN1v;

  def("dcopy_", (void (*)(const aN&  , aN&, int, int))&array::dcopy_<T,N>,
      (arg("source"), arg("dest"), arg("dim"), arg("pos")), DCOPY_DOC_);
  def("dcopy" , (void (*)(const aN&  , aN&, int, int))&array::dcopy<T,N>,
      (arg("source"), arg("dest"), arg("dim"), arg("pos")), DCOPY_DOC);

  def("dcopy_", (void (*)(const aN1& , aN&, int, int))&array::dcopy_<T,N>,
      (arg("source"), arg("dest"), arg("dim"), arg("pos")), DCOPY_DOC1_);
  def("dcopy" , (void (*)(const aN1& , aN&, int, int))&array::dcopy<T,N>,
      (arg("source"), arg("dest"), arg("dim"), arg("pos")), DCOPY_DOC1);

  def("cat_"  , (void (*)(const aNv& , aN&, int))&array::cat_<T,N>,
      (arg("source"), arg("dest"), arg("dim")), CAT_DOC_);
  def("cat"   , (void (*)(const aNv& , aN&, int))&array::cat<T,N>,
      (arg("source"), arg("dest"), arg("dim")), CAT_DOC);

  def("cat"   , (aN   (*)(const aNv& , int))&cat<T,N>,
      (arg("source"), arg("dim")), CAT_DOC_PY);

  def("stack_"  , (void (*)(const aN1v&, aN&))&array::stack_<T,N>,
      (arg("source"), arg("dest")), STACK_DOC_);
  def("stack"   , (void (*)(const aN1v&, aN&))&array::stack<T,N>,
      (arg("source"), arg("dest")), STACK_DOC);

  def("stack"  , (aN   (*)(const aN1v&))&stack<T,N>,
      (arg("source")), STACK_DOC_PY);

}

void bind_array_cat () {
  bind_cat1<bool>(); 
  bind_cat1<int8_t>();
  bind_cat1<int16_t>();
  bind_cat1<int32_t>();
  bind_cat1<int64_t>();
  bind_cat1<uint8_t>();
  bind_cat1<uint16_t>();
  bind_cat1<uint32_t>();
  bind_cat1<uint64_t>();
  bind_cat1<float>();
  bind_cat1<double>();
  bind_cat1<std::complex<float> >();
  bind_cat1<std::complex<double> >();

# define BOOST_PP_LOCAL_LIMITS (2, TORCH_MAX_DIM)
# define BOOST_PP_LOCAL_MACRO(D) \
  bind_cat<bool,D>(); \
  bind_cat<int8_t,D>();\
  bind_cat<int16_t,D>();\
  bind_cat<int32_t,D>();\
  bind_cat<int64_t,D>();\
  bind_cat<uint8_t,D>();\
  bind_cat<uint16_t,D>();\
  bind_cat<uint32_t,D>();\
  bind_cat<uint64_t,D>();\
  bind_cat<float,D>();\
  bind_cat<double,D>();\
  bind_cat<std::complex<float>,D>();\
  bind_cat<std::complex<double>,D>();
# include BOOST_PP_LOCAL_ITERATE()
}
