/**
 * @file python/ip/src/histo.cc
 * @date Mon Apr 18 16:08:34 2011 +0200
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 *
 * @brief Binds histogram to python
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "core/python/ndarray.h"
#include "ip/histo.h"

using namespace boost::python;
namespace ip = Torch::ip;
namespace tp = Torch::python;
namespace ca = Torch::core::array;

template <typename T>
static object inner_histo1 (tp::const_ndarray input) {
  int size = Torch::ip::detail::getHistoSize<T>();
  tp::ndarray out(ca::t_uint64, size);
  blitz::Array<uint64_t,1> out_ = out.bz<uint64_t,1>();
  ip::histogram(input.bz<T,2>(), out_, false);
  return out.self();
}

static object histo1 (tp::const_ndarray input) {
  const ca::typeinfo& info = input.type();
  switch (info.dtype) {
    case ca::t_uint8: return inner_histo1<uint8_t>(input);
    case ca::t_uint16: return inner_histo1<uint16_t>(input);
    default:
      PYTHON_ERROR(TypeError, "unsupported histogram operation for type '%s'", info.str().c_str());
  }
}

template <typename T>
static void inner_histo2 (tp::const_ndarray input, tp::ndarray output,
    bool accumulate) {
  blitz::Array<uint64_t,1> out_ = output.bz<uint64_t,1>();
  ip::histogram(input.bz<T,2>(), out_, accumulate);
}

static void histo2 (tp::const_ndarray input, tp::ndarray output,
    bool accumulate=false) {
  const ca::typeinfo& info = input.type();
  switch (info.dtype) {
    case ca::t_uint8: return inner_histo2<uint8_t>(input, output, accumulate);
    case ca::t_uint16: return inner_histo2<uint16_t>(input, output, accumulate);
    default:
      PYTHON_ERROR(TypeError, "unsupported histogram operation for type '%s'", info.str().c_str());
  }
}

BOOST_PYTHON_FUNCTION_OVERLOADS(histo2_overloads, histo2, 2, 3)

template <typename T>
static void inner_histo3 (tp::const_ndarray input, tp::ndarray output,
    object max, bool accumulate) {
  blitz::Array<uint64_t,1> out_ = output.bz<uint64_t,1>();
  T tmax = extract<T>(max);
  ip::histogram(input.bz<T,2>(), out_, (T)0, tmax, (uint32_t)(tmax+1), accumulate);
}

static void histo3 (tp::const_ndarray input, tp::ndarray output, object max,
    bool accumulate=false) {
  const ca::typeinfo& info = input.type();
  switch (info.dtype) {
    case ca::t_int8: 
      return inner_histo3<int8_t>(input, output, max, accumulate);
    case ca::t_int16: 
      return inner_histo3<int16_t>(input, output, max, accumulate);
    case ca::t_int32: 
      return inner_histo3<int32_t>(input, output, max, accumulate);
    case ca::t_int64: 
      return inner_histo3<int64_t>(input, output, max, accumulate);
    case ca::t_uint8: 
      return inner_histo3<uint8_t>(input, output, max, accumulate);
    case ca::t_uint16: 
      return inner_histo3<uint16_t>(input, output, max, accumulate);
    case ca::t_uint32: 
      return inner_histo3<uint32_t>(input, output, max, accumulate);
    case ca::t_uint64: 
      return inner_histo3<uint64_t>(input, output, max, accumulate);
    case ca::t_float32: 
      return inner_histo3<float>(input, output, max, accumulate);
    case ca::t_float64: 
      return inner_histo3<double>(input, output, max, accumulate);
    default:
      PYTHON_ERROR(TypeError, "unsupported histogram operation for type '%s'", info.str().c_str());
  }
}

BOOST_PYTHON_FUNCTION_OVERLOADS(histo3_overloads, histo3, 3, 4)

template <typename T>
static void inner_histo4 (tp::const_ndarray input, tp::ndarray output,
    object min, object max, bool accumulate) {
  blitz::Array<uint64_t,1> out_ = output.bz<uint64_t,1>();
  T tmin = extract<T>(min);
  T tmax = extract<T>(max);
  ip::histogram(input.bz<T,2>(), out_, tmin, tmax, (uint32_t)(tmax-tmin+1), accumulate);
}

static void histo4 (tp::const_ndarray input, tp::ndarray output, 
    object min, object max, bool accumulate=false) {
  const ca::typeinfo& info = input.type();
  switch (info.dtype) {
    case ca::t_int8: 
      return inner_histo4<int8_t>(input, output, min, max, accumulate);
    case ca::t_int16: 
      return inner_histo4<int16_t>(input, output, min, max, accumulate);
    case ca::t_int32: 
      return inner_histo4<int32_t>(input, output, min, max, accumulate);
    case ca::t_int64: 
      return inner_histo4<int64_t>(input, output, min, max, accumulate);
    case ca::t_uint8: 
      return inner_histo4<uint8_t>(input, output, min, max, accumulate);
    case ca::t_uint16: 
      return inner_histo4<uint16_t>(input, output, min, max, accumulate);
    case ca::t_uint32: 
      return inner_histo4<uint32_t>(input, output, min, max, accumulate);
    case ca::t_uint64: 
      return inner_histo4<uint64_t>(input, output, min, max, accumulate);
    case ca::t_float32: 
      return inner_histo4<float>(input, output, min, max, accumulate);
    case ca::t_float64: 
      return inner_histo4<double>(input, output, min, max, accumulate);
    default:
      PYTHON_ERROR(TypeError, "unsupported histogram operation for type '%s'", info.str().c_str());
  }
}

BOOST_PYTHON_FUNCTION_OVERLOADS(histo4_overloads, histo4, 4, 5)

template <typename T>
static void inner_histo5 (tp::const_ndarray input, tp::ndarray output,
    object min, object max, uint32_t nbins, bool accumulate) {
  blitz::Array<uint64_t,1> out_ = output.bz<uint64_t,1>();
  T tmin = extract<T>(min);
  T tmax = extract<T>(max);
  ip::histogram(input.bz<T,2>(), out_, tmin, tmax, nbins, accumulate);
}

static void histo5 (tp::const_ndarray input, tp::ndarray output, 
    object min, object max, uint32_t nbins, bool accumulate=false) {
  const ca::typeinfo& info = input.type();
  switch (info.dtype) {
    case ca::t_int8: 
      return inner_histo5<int8_t>(input, output, min, max, nbins, accumulate);
    case ca::t_int16: 
      return inner_histo5<int16_t>(input, output, min, max, nbins, accumulate);
    case ca::t_int32: 
      return inner_histo5<int32_t>(input, output, min, max, nbins, accumulate);
    case ca::t_int64: 
      return inner_histo5<int64_t>(input, output, min, max, nbins, accumulate);
    case ca::t_uint8: 
      return inner_histo5<uint8_t>(input, output, min, max, nbins, accumulate);
    case ca::t_uint16: 
      return inner_histo5<uint16_t>(input, output, min, max, nbins, accumulate);
    case ca::t_uint32: 
      return inner_histo5<uint32_t>(input, output, min, max, nbins, accumulate);
    case ca::t_uint64: 
      return inner_histo5<uint64_t>(input, output, min, max, nbins, accumulate);
    case ca::t_float32: 
      return inner_histo5<float>(input, output, min, max, nbins, accumulate);
    case ca::t_float64: 
      return inner_histo5<double>(input, output, min, max, nbins, accumulate);
    default:
      PYTHON_ERROR(TypeError, "unsupported histogram operation for type '%s'", info.str().c_str());
  }
}

BOOST_PYTHON_FUNCTION_OVERLOADS(histo5_overloads, histo5, 5, 6)

template <typename T>
static object inner_histo3a (tp::const_ndarray input, object max) {
  T tmax = extract<T>(max);
  uint32_t size = (uint32_t)(tmax + 1);
  tp::ndarray out(ca::t_uint64, size);
  blitz::Array<uint64_t,1> out_ = out.bz<uint64_t,1>();
  ip::histogram(input.bz<T,2>(), out_, (T)0, tmax, size, false);
  return out.self();
}

static object histo3a (tp::const_ndarray input, object max) {
  const ca::typeinfo& info = input.type();
  switch (info.dtype) {
    case ca::t_int8: return inner_histo3a<int8_t>(input, max);
    case ca::t_int16: return inner_histo3a<int16_t>(input, max);
    case ca::t_int32: return inner_histo3a<int32_t>(input, max);
    case ca::t_int64: return inner_histo3a<int64_t>(input, max);
    case ca::t_uint8: return inner_histo3a<uint8_t>(input, max);
    case ca::t_uint16: return inner_histo3a<uint16_t>(input, max);
    case ca::t_uint32: return inner_histo3a<uint32_t>(input, max);
    case ca::t_uint64: return inner_histo3a<uint64_t>(input, max);
    case ca::t_float32: return inner_histo3a<float>(input, max);
    case ca::t_float64: return inner_histo3a<double>(input, max);
    default:
      PYTHON_ERROR(TypeError, "unsupported histogram operation for type '%s'", info.str().c_str());
  }
}

template <typename T>
static object inner_histo4a (tp::const_ndarray input, object min, object max) {
  T tmin = extract<T>(min);
  T tmax = extract<T>(max);
  int64_t size = (int64_t)(tmax - tmin + 1);
  tp::ndarray out(ca::t_uint64, size);
  blitz::Array<uint64_t,1> out_ = out.bz<uint64_t,1>();
  ip::histogram(input.bz<T,2>(), out_, tmin, tmax, size, false);
  return out.self();
}

static object histo4a (tp::const_ndarray input, object min, object max) {
  const ca::typeinfo& info = input.type();
  switch (info.dtype) {
    case ca::t_int8: 
      return inner_histo4a<int8_t>(input, min, max);
    case ca::t_int16: 
      return inner_histo4a<int16_t>(input, min, max);
    case ca::t_int32: 
      return inner_histo4a<int32_t>(input, min, max);
    case ca::t_int64: 
      return inner_histo4a<int64_t>(input, min, max);
    case ca::t_uint8: 
      return inner_histo4a<uint8_t>(input, min, max);
    case ca::t_uint16: 
      return inner_histo4a<uint16_t>(input, min, max);
    case ca::t_uint32: 
      return inner_histo4a<uint32_t>(input, min, max);
    case ca::t_uint64: 
      return inner_histo4a<uint64_t>(input, min, max);
    case ca::t_float32: 
      return inner_histo4a<float>(input, min, max);
    case ca::t_float64: 
      return inner_histo4a<double>(input, min, max);
    default:
      PYTHON_ERROR(TypeError, "unsupported histogram operation for type '%s'", info.str().c_str());
  }
}

template <typename T>
static object inner_histo5a (tp::const_ndarray input, object min, object max,
    uint32_t nbins) {
  T tmin = extract<T>(min);
  T tmax = extract<T>(max);
  tp::ndarray out(ca::t_uint64, nbins);
  blitz::Array<uint64_t,1> out_ = out.bz<uint64_t,1>();
  ip::histogram(input.bz<T,2>(), out_, tmin, tmax, nbins, false);
  return out.self();
}

static object histo5a (tp::const_ndarray input, 
    object min, object max, uint32_t nbins) {
  const ca::typeinfo& info = input.type();
  switch (info.dtype) {
    case ca::t_int8: 
      return inner_histo5a<int8_t>(input, min, max, nbins);
    case ca::t_int16: 
      return inner_histo5a<int16_t>(input, min, max, nbins);
    case ca::t_int32: 
      return inner_histo5a<int32_t>(input, min, max, nbins);
    case ca::t_int64: 
      return inner_histo5a<int64_t>(input, min, max, nbins);
    case ca::t_uint8: 
      return inner_histo5a<uint8_t>(input, min, max, nbins);
    case ca::t_uint16: 
      return inner_histo5a<uint16_t>(input, min, max, nbins);
    case ca::t_uint32: 
      return inner_histo5a<uint32_t>(input, min, max, nbins);
    case ca::t_uint64: 
      return inner_histo5a<uint64_t>(input, min, max, nbins);
    case ca::t_float32: 
      return inner_histo5a<float>(input, min, max, nbins);
    case ca::t_float64: 
      return inner_histo5a<double>(input, min, max, nbins);
    default:
      PYTHON_ERROR(TypeError, "unsupported histogram operation for type '%s'", info.str().c_str());
  }
}

void bind_ip_histogram() {
  def("histogram_", &histo2, histo2_overloads((arg("src"), arg("histo"), arg("accumulate")=false), "Compute an histogram of a 2D array. The histogram must have a size of 2^N-1 elements, where N is the number of bits in input. If the accumulate flag is set (defaults to False), then I accumulate instead of resetting the histogram."));
  
  def("histogram_", &histo3, histo3_overloads((arg("src"), arg("histo"), arg("max"), arg("accumulate")=false), "Compute an histogram of a 2D array.\nsrc elements are in range [0, max] (max >= 0)\nhisto must have a size of max elements"));
  
  def("histogram_", &histo4, histo4_overloads((arg("src"), arg("histo"), arg("min"), arg("max"), arg("accumulate")=false), "Compute an histogram of a 2D array.\nsrc elements are in range [min, max] (max >= min)\nhisto must have a size of max-min elements"));
  
  def("histogram_", &histo5, histo5_overloads((arg("src"), arg("histo"), arg("min"), arg("max"), arg("nb_bins"), arg("accumulate")=false), "Compute an histogram of a 2D array.\nsrc elements are in range [min, max] (max >= min)\nhisto must have a size of nb_bins elements")); 
  
  def("histogram", &histo1, (args("src")), "Compute an histogram of a 2D array");
  def("histogram", &histo3a, (arg("src"), arg("max")), "Return an histogram of a 2D array.\nsrc elements are in range [0, max] (max >= 0)\n");
  
  def("histogram", &histo4a, (arg("src"), arg("min"), arg("max")), "Return an histogram of a 2D array.\nsrc elements are in range [min, max] (max >= min)\n");
  
  def("histogram", &histo5a, (arg("src"), arg("min"), arg("max"), arg("nb_bins")), "Return an histogram of a 2D array.\nsrc elements are in range [min, max] (max >= min)\n");
}
