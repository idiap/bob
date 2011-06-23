/**
 * @author <a href="mailto:niklas.johansson@idiap.ch">Niklas Johansson</a>
 * @brief Makes a 1D array from a 2D array by concatenating its lines.
 */

#include <boost/python.hpp>
#include <blitz/array.h>

#include "core/array_assert.h"
#include "core/python/array_base.h"

namespace bp = boost::python;
namespace tp = Torch::python;

template <typename T> static
void as_row2(const blitz::Array<T,2>& from, blitz::Array<T,1>& to) {
  Torch::core::array::assertSameDimensionLength(from.size(), to.size());
  blitz::Range a = blitz::Range::all();
  for (int i=0; i<from.rows(); ++i) {
    to(blitz::Range(i*from.cols(), (i+1)*from.cols()-1)) = from(i,a);
  }
}

template <typename T> static
blitz::Array<T,1> as_row1(const blitz::Array<T,2>& from) {
  blitz::Array<T,1> to(from.size());
  as_row2(from, to);
  return to;
}

template <typename T> static void create_as_one_row(tp::array<T,2>& array) {
	array.object()->def("as_row", &as_row1<T>, "This method will take a 2D blitz array and turn it into a 1D blitz array by concatenating all of its rows, in order.");
	array.object()->def("as_row", &as_row2<T>, "This method will take a 2D blitz array and turn it into a 1D blitz array by concatenating all of its rows, in order.");
}

void bind_as_one_row() {
  create_as_one_row(tp::bool_2);
  create_as_one_row(tp::int8_2);
  create_as_one_row(tp::int16_2);
  create_as_one_row(tp::int32_2);
  create_as_one_row(tp::int64_2);
  create_as_one_row(tp::uint8_2);
  create_as_one_row(tp::uint16_2);
  create_as_one_row(tp::uint32_2);
  create_as_one_row(tp::uint64_2);
  create_as_one_row(tp::float32_2);
  create_as_one_row(tp::float64_2);
  create_as_one_row(tp::complex64_2);
  create_as_one_row(tp::complex128_2);
}
