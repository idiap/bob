/**
 * @file math/python/stats.cc
 * @date Mon Jun 20 11:47:58 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Python bindings to statistical methods
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

#include <boost/python.hpp>
#include <bob/python/ndarray.h>
#include <boost/python/stl_iterator.hpp>
#include <bob/math/stats.h>

using namespace boost::python;

static const char* SCATTER_DOC1 = "Computes the scatter matrix of a 2D array considering data is organized row-wise (each sample is a row, each feature is a column). The resulting matrix 's' has to be square with extents equal to the number of columns in a.";

static const char* SCATTER_DOC2 = "Computes the scatter matrix of a 2D array considering data is organized row-wise (each sample is a row, each feature is a column). This variant also returns the sample means in 'm'. The resulting arrays 'm' and 's' have to have the correct sizes (s should be square with extents equal to the number of columns in a and m should be a 1D vector with extents equal to the number of columns in a).";

static const char* SCATTER_DOC3 = "Computes the scatter matrix of a 2D array considering data is organized row-wise (each sample is a row, each feature is a column). This variant returns the sample means and the scatter matrix in a tuple. If you are looking for efficiency, prefer the variants that receive the output variable as one of the input parameters. This version will allocate the resulting arrays 'm' and 's' internally every time it is called.";


static const char* SCATTERS_DOC1 = "Computes the within-class and between-class scatter matrices of a set of 2D arrays considering data is organized row-wise (each sample is a row, each feature is a column). This implies that all the 2D arrays in 'data' should have the same number of columns. The resulting matrices 'sb' and 'sw' have to be square with extents equal to the number of columns in the arrays of 'data'.";

static const char* SCATTERS_DOC2 = "Computes the within-class and between-class scatter matrices of a set of 2D arrays considering data is organized row-wise (each sample is a row, each feature is a column). This implies that all the 2D arrays in 'data' should have the same number of columns. This variant also returns the total sample means in 'm'. The resulting arrays 'm', 'sb' and 'sw' have to have the correct sizes ('sb' and 'sw' should be square with extents equal to the number of columns in the arrays of 'data' and 'm' should be a 1D vector with extents equal to the number of columns in the arrays of 'data').";

static const char* SCATTERS_DOC3 = "Computes the within-class and between-class scatter matrices of a set of 2D arrays considering data is organized row-wise (each sample is a row, each feature is a column). This implies that all the 2D arrays in 'data' should have the same number of columns. This variant returns the sample means and the scatter matrices 'sb' and 'sw' in a tuple. If you are looking for efficiency, prefer the variants that receive the output variable as one of the input parameters. This variant will allocate the resulting arrays 'm' and 'sb' and 'sw' internally every time it is called.";


template <typename T> static tuple scatter_inner(bob::python::const_ndarray A) {
  const bob::core::array::typeinfo& info = A.type();
  bob::python::ndarray S(info.dtype, info.shape[1], info.shape[1]);
  blitz::Array<T,2> S_ = S.bz<T,2>();
  bob::python::ndarray M(info.dtype, info.shape[1]);
  blitz::Array<T,1> M_ = M.bz<T,1>();
  bob::math::scatter(A.bz<T,2>(), S_, M_);
  return make_tuple(S,M);
}

static tuple scatter(bob::python::const_ndarray A) {
  const bob::core::array::typeinfo& info = A.type();
  switch (info.dtype) {
    case bob::core::array::t_float32:
      return scatter_inner<float>(A);
    case bob::core::array::t_float64:
      return scatter_inner<double>(A);
    default:
      PYTHON_ERROR(TypeError, "scatter matrix computation does not support '%s'", info.str().c_str());
  }
}

template <typename T>
static void scatter_nocheck_inner(bob::python::const_ndarray A, bob::python::ndarray S) {
  blitz::Array<T,2> S_ = S.bz<T,2>();
  bob::math::scatter_<T>(A.bz<T,2>(), S_);
}

static void scatter_nocheck(bob::python::const_ndarray A, bob::python::ndarray S) {
  const bob::core::array::typeinfo& info = A.type();
  switch (info.dtype) {
    case bob::core::array::t_float32:
      return scatter_nocheck_inner<float>(A, S);
    case bob::core::array::t_float64:
      return scatter_nocheck_inner<double>(A, S);
    default:
      PYTHON_ERROR(TypeError, "(unchecked) scatter matrix computation does not support '%s'", info.str().c_str());
  }
}

template <typename T>
static void scatter_check_inner(bob::python::const_ndarray A, bob::python::ndarray S) {
  blitz::Array<T,2> S_ = S.bz<T,2>();
  bob::math::scatter<T>(A.bz<T,2>(), S_);
}

static void scatter_check(bob::python::const_ndarray A, bob::python::ndarray S) {
  const bob::core::array::typeinfo& info = A.type();
  switch (info.dtype) {
    case bob::core::array::t_float32:
      return scatter_check_inner<float>(A, S);
    case bob::core::array::t_float64:
      return scatter_check_inner<double>(A, S);
    default:
      PYTHON_ERROR(TypeError, "scatter matrix computation does not support '%s'", info.str().c_str());
  }
}

template <typename T>
static void scatter_M_nocheck_inner(bob::python::const_ndarray A, bob::python::ndarray S,
    bob::python::ndarray M) {
  blitz::Array<T,2> S_ = S.bz<T,2>();
  blitz::Array<T,1> M_ = M.bz<T,1>();
  bob::math::scatter_<T>(A.bz<T,2>(), S_, M_);
}

static void scatter_M_nocheck(bob::python::const_ndarray A, bob::python::ndarray S,
    bob::python::ndarray M) {
  const bob::core::array::typeinfo& info = A.type();
  switch (info.dtype) {
    case bob::core::array::t_float32:
      return scatter_M_nocheck_inner<float>(A, S, M);
    case bob::core::array::t_float64:
      return scatter_M_nocheck_inner<double>(A, S, M);
    default:
      PYTHON_ERROR(TypeError, "(unchecked) scatter matrix computation does not support '%s'", info.str().c_str());
  }
}

template <typename T>
static void scatter_M_check_inner(bob::python::const_ndarray A, bob::python::ndarray S,
    bob::python::ndarray M) {
  blitz::Array<T,2> S_ = S.bz<T,2>();
  blitz::Array<T,1> M_ = M.bz<T,1>();
  bob::math::scatter<T>(A.bz<T,2>(), S_, M_);
}

static void scatter_M_check(bob::python::const_ndarray A, bob::python::ndarray S,
    bob::python::ndarray M) {
  const bob::core::array::typeinfo& info = A.type();
  switch (info.dtype) {
    case bob::core::array::t_float32:
      return scatter_M_check_inner<float>(A, S, M);
    case bob::core::array::t_float64:
      return scatter_M_check_inner<double>(A, S, M);
    default:
      PYTHON_ERROR(TypeError, "scatter matrix computation does not support '%s'", info.str().c_str());
  }
}


template <typename T> 
static tuple scatters_inner(object data, bob::python::const_ndarray sample) 
{
  stl_input_iterator<blitz::Array<T,2> > dbegin(data), dend;
  std::vector<blitz::Array<T,2> > vdata(dbegin, dend);
  const bob::core::array::typeinfo& info = sample.type();
  bob::python::ndarray Sb(info.dtype, info.shape[1], info.shape[1]);
  blitz::Array<T,2> Sb_ = Sb.bz<T,2>();
  bob::python::ndarray Sw(info.dtype, info.shape[1], info.shape[1]);
  blitz::Array<T,2> Sw_ = Sw.bz<T,2>();
  bob::python::ndarray M(info.dtype, info.shape[1]);
  blitz::Array<T,1> M_ = M.bz<T,1>();
  bob::math::scatters(vdata, Sw_, Sb_, M_);
  return make_tuple(Sw, Sb, M);
}

static tuple scatters(object data) {
  stl_input_iterator<bob::python::const_ndarray > dbegin(data), dend;
  std::vector<bob::python::const_ndarray > vdata(dbegin, dend);
  const bob::core::array::typeinfo& info = vdata[0].type();
  switch (info.dtype) {
    case bob::core::array::t_float32:
      return scatters_inner<float>(data, vdata[0]);
    case bob::core::array::t_float64:
      return scatters_inner<double>(data, vdata[0]);
    default:
      PYTHON_ERROR(TypeError, "scatters matrix computation does not support '%s'", info.str().c_str());
  }
}

template <typename T>
static void scatters_nocheck_inner(const std::vector<blitz::Array<T,2> >& data,
  bob::python::ndarray Sw, bob::python::ndarray Sb)
{
  blitz::Array<T,2> Sw_ = Sw.bz<T,2>();
  blitz::Array<T,2> Sb_ = Sb.bz<T,2>();
  bob::math::scatters_<T>(data, Sw_, Sb_);
}

static void scatters_nocheck(object data, bob::python::ndarray Sw,
    bob::python::ndarray Sb)
{
  const bob::core::array::typeinfo& info = Sw.type();
  switch (info.dtype) {
    case bob::core::array::t_float32:
    {  
      stl_input_iterator<blitz::Array<float,2> > dbegin(data), dend;
      std::vector<blitz::Array<float,2> > vdata(dbegin, dend);
      return scatters_nocheck_inner<float>(vdata, Sw, Sb);
    }
    case bob::core::array::t_float64:
    {  
      stl_input_iterator<blitz::Array<double,2> > dbegin(data), dend;
      std::vector<blitz::Array<double,2> > vdata(dbegin, dend);
      return scatters_nocheck_inner<double>(vdata, Sw, Sb);
    }
    default:
      PYTHON_ERROR(TypeError, "(unchecked) scatters matrix computation does not support '%s'", info.str().c_str());
  }
}

template <typename T>
static void scatters_check_inner(const std::vector<blitz::Array<T,2> >& data,
  bob::python::ndarray Sw, bob::python::ndarray Sb)
{
  blitz::Array<T,2> Sw_ = Sw.bz<T,2>();
  blitz::Array<T,2> Sb_ = Sb.bz<T,2>();
  bob::math::scatters<T>(data, Sw_, Sb_);
}

static void scatters_check(object data, bob::python::ndarray Sw,
    bob::python::ndarray Sb)
{
  const bob::core::array::typeinfo& info = Sw.type();
  switch (info.dtype) {
    case bob::core::array::t_float32:
    {  
      stl_input_iterator<blitz::Array<float,2> > dbegin(data), dend;
      std::vector<blitz::Array<float,2> > vdata(dbegin, dend);
      return scatters_check_inner<float>(vdata, Sw, Sb);
    }
    case bob::core::array::t_float64:
    {  
      stl_input_iterator<blitz::Array<double,2> > dbegin(data), dend;
      std::vector<blitz::Array<double,2> > vdata(dbegin, dend);
      return scatters_check_inner<double>(vdata, Sw, Sb);
    }
    default:
      PYTHON_ERROR(TypeError, "scatters matrix computation does not support '%s'", info.str().c_str());
  }
}

template <typename T>
static void scatters_M_nocheck_inner(const std::vector<blitz::Array<T,2> >& data, 
  bob::python::ndarray Sw, bob::python::ndarray Sb, bob::python::ndarray M)
{
  blitz::Array<T,2> Sw_ = Sw.bz<T,2>();
  blitz::Array<T,2> Sb_ = Sb.bz<T,2>();
  blitz::Array<T,1> M_ = M.bz<T,1>();
  bob::math::scatters_<T>(data, Sw_, Sb_, M_);
}

static void scatters_M_nocheck(object data, bob::python::ndarray Sw,
    bob::python::ndarray Sb, bob::python::ndarray M)
{
  const bob::core::array::typeinfo& info = Sw.type();
  switch (info.dtype) {
    case bob::core::array::t_float32:
    {  
      stl_input_iterator<blitz::Array<float,2> > dbegin(data), dend;
      std::vector<blitz::Array<float,2> > vdata(dbegin, dend);
      return scatters_M_nocheck_inner<float>(vdata, Sw, Sb, M);
    }
    case bob::core::array::t_float64:
    {  
      stl_input_iterator<blitz::Array<double,2> > dbegin(data), dend;
      std::vector<blitz::Array<double,2> > vdata(dbegin, dend);
      return scatters_M_nocheck_inner<double>(vdata, Sw, Sb, M);
    }
    default:
      PYTHON_ERROR(TypeError, "(unchecked) scatters matrix computation does not support '%s'", info.str().c_str());
  }
}

template <typename T>
static void scatters_M_check_inner(const std::vector<blitz::Array<T,2> >& data,
  bob::python::ndarray Sw, bob::python::ndarray Sb, bob::python::ndarray M)
{
  blitz::Array<T,2> Sw_ = Sw.bz<T,2>();
  blitz::Array<T,2> Sb_ = Sb.bz<T,2>();
  blitz::Array<T,1> M_ = M.bz<T,1>();
  bob::math::scatters<T>(data, Sw_, Sb_, M_);
}

static void scatters_M_check(object data, bob::python::ndarray Sw,
  bob::python::ndarray Sb, bob::python::ndarray M)
{
  const bob::core::array::typeinfo& info = Sw.type();
  switch (info.dtype) {
    case bob::core::array::t_float32:
    {  
      stl_input_iterator<blitz::Array<float,2> > dbegin(data), dend;
      std::vector<blitz::Array<float,2> > vdata(dbegin, dend);
      return scatters_M_check_inner<float>(vdata, Sw, Sb, M);
    }
    case bob::core::array::t_float64:
    {  
      stl_input_iterator<blitz::Array<double,2> > dbegin(data), dend;
      std::vector<blitz::Array<double,2> > vdata(dbegin, dend);
      return scatters_M_check_inner<double>(vdata, Sw, Sb, M);
    }
    default:
      PYTHON_ERROR(TypeError, "scatters matrix computation does not support '%s'", info.str().c_str());
  }
}


void bind_math_stats() {
  def("scatter_", &scatter_nocheck, (arg("a"), arg("s")), SCATTER_DOC1);
  def("scatter", &scatter_check, (arg("a"), arg("s")), SCATTER_DOC1);
  
  def("scatter_", &scatter_M_nocheck, (arg("a"), arg("s"), arg("m")), SCATTER_DOC2);
  def("scatter", &scatter_M_check, (arg("a"), arg("s"), arg("m")), SCATTER_DOC2);

  def("scatter", &scatter, (arg("a")), SCATTER_DOC3);

  def("scatters_", &scatters_nocheck, (arg("data"), arg("sw"), arg("sb")), SCATTERS_DOC1);
  def("scatters", &scatters_check, (arg("data"), arg("sw"), arg("sb")), SCATTERS_DOC1);
  
  def("scatters_", &scatters_M_nocheck, (arg("data"), arg("sw"), arg("sb"), arg("m")), SCATTERS_DOC2);
  def("scatters", &scatters_M_check, (arg("data"), arg("sw"), arg("sb"), arg("m")), SCATTERS_DOC2);

  def("scatters", &scatters, (arg("data")), SCATTERS_DOC3);
}
