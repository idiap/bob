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
cat(const std::vector<blitz::Array<T,N-1> >& source, int D) {
  if (!source.size()) return blitz::Array<T,N>();
  blitz::TinyVector<int,N> shape;
  for (int k=0, l=0; k<N; ++k) {
    if (k == D) shape(k) = 0;
    else shape(l++) = source[0].shape()(k);
  }
  for (size_t i=0; i<source.size(); ++i) shape(D) += source[i].extent(D);
  blitz::Array<T,N> dest(shape);
  array::cat(source, dest, D);
  return dest;
}

template <typename T> static void bind_cat1() {
  typedef blitz::Array<T,1> aN;
  typedef std::vector<aN> aNv;

  def("dcopy_", (void (*)(const aN&,  aN&, int, int))&array::dcopy_<T,1>);
  def("dcopy" , (void (*)(const aN&,  aN&, int, int))&array::dcopy<T,1>);

  def("cat_"  , (void (*)(const aNv&, aN&, int))&array::cat_<T,1>);
  def("cat"   , (void (*)(const aNv&, aN&, int))&array::cat<T,1>);

  def("cat"   , (aN   (*)(const aNv&, int))&cat<T,1>);

}

template <typename T, int N> static void bind_cat() {
  typedef blitz::Array<T,N> aN;
  typedef std::vector<aN> aNv;
  typedef blitz::Array<T,N-1> aN1;
  typedef std::vector<aN1> aN1v;

  def("dcopy_", (void (*)(const aN&  , aN&, int, int))&array::dcopy_<T,N>);
  def("dcopy" , (void (*)(const aN&  , aN&, int, int))&array::dcopy<T,N>);

  def("dcopy_", (void (*)(const aN1& , aN&, int, int))&array::dcopy_<T,N>);
  def("dcopy" , (void (*)(const aN1& , aN&, int, int))&array::dcopy<T,N>);

  def("cat_"  , (void (*)(const aNv& , aN&, int))&array::cat_<T,N>);
  def("cat"   , (void (*)(const aNv& , aN&, int))&array::cat<T,N>);

  def("cat_"  , (void (*)(const aN1v&, aN&, int))&array::cat_<T,N>);
  def("cat"   , (void (*)(const aN1v&, aN&, int))&array::cat<T,N>);

  def("cat"   , (aN   (*)(const aNv& , int))&cat<T,N>);
  def("pcat"  , (aN   (*)(const aN1v&, int))&cat<T,N>);

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
