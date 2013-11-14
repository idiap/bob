/**
 * @file sp/python/dct.cc
 * @date Thu Apr 14 13:39:40 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds the (fast) Discrete Cosine Transform to python.
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

#include <bob/python/ndarray.h>

#include <bob/sp/DCT1DKiss.h>
#include <bob/sp/DCT2DKiss.h>


using namespace boost::python;

// documentation for classes
static const char* DCT1DKiss_DOC = "Objects of this class, after configuration, can compute the direct DCT of a 1D array/signal.";
static const char* IDCT1DKiss_DOC = "Objects of this class, after configuration, can compute the inverse DCT of a 1D array/signal.";
static const char* DCT2DKiss_DOC = "Objects of this class, after configuration, can compute the direct DCT of a 2D array/signal.";
static const char* IDCT2DKiss_DOC = "Objects of this class, after configuration, can compute the inverse DCT of a 2D array/signal.";

// free methods documentation
static const char* DCT_DOC = "Compute the direct DCT of a 1 or 2D array/signal of type float64.";
static const char* IDCT_DOC = "Compute the inverse DCT of a 1 or 2D array/signal of type float64.";


static void py_dct1d_c(bob::sp::DCT1DKiss& op, bob::python::const_ndarray src,
  bob::python::ndarray dst) 
{
  blitz::Array<double,1> dst_ = dst.bz<double,1>();
  op(src.bz<double,1>(), dst_);
}

static object py_dct1d_p(bob::sp::DCT1DKiss& op, bob::python::const_ndarray src)
{
  bob::python::ndarray dst(bob::core::array::t_float64, op.getLength());
  blitz::Array<double,1> dst_ = dst.bz<double,1>();
  op(src.bz<double,1>(), dst_);
  return dst.self();
}

static void py_idct1d_c(bob::sp::IDCT1DKiss& op, bob::python::const_ndarray src,
  bob::python::ndarray dst) 
{
  blitz::Array<double,1> dst_ = dst.bz<double,1>();
  op(src.bz<double,1>(), dst_);
}

static object py_idct1d_p(bob::sp::IDCT1DKiss& op, bob::python::const_ndarray src)
{
  bob::python::ndarray dst(bob::core::array::t_float64, op.getLength());
  blitz::Array<double,1> dst_ = dst.bz<double,1>();
  op(src.bz<double,1>(), dst_);
  return dst.self();
}


static void py_dct2d_c(bob::sp::DCT2DKiss& op, bob::python::const_ndarray src,
  bob::python::ndarray dst) 
{
  blitz::Array<double,2> dst_ = dst.bz<double,2>();
  op(src.bz<double,2>(), dst_);
}

static object py_dct2d_p(bob::sp::DCT2DKiss& op, bob::python::const_ndarray src)
{
  bob::python::ndarray dst(bob::core::array::t_float64, op.getHeight(), 
    op.getWidth());
  blitz::Array<double,2> dst_ = dst.bz<double,2>();
  op(src.bz<double,2>(), dst_);
  return dst.self();
}

static void py_idct2d_c(bob::sp::IDCT2DKiss& op, bob::python::const_ndarray src,
  bob::python::ndarray dst) 
{
  blitz::Array<double,2> dst_ = dst.bz<double,2>();
  op(src.bz<double,2>(), dst_);
}

static object py_idct2d_p(bob::sp::IDCT2DKiss& op, bob::python::const_ndarray src)
{
  bob::python::ndarray dst(bob::core::array::t_float64, op.getHeight(), 
    op.getWidth());
  blitz::Array<double,2> dst_ = dst.bz<double,2>();
  op(src.bz<double,2>(), dst_);
  return dst.self();
}


static object script_dct(bob::python::const_ndarray ar) 
{
  const bob::core::array::typeinfo& info = ar.type();
  bob::python::ndarray res(info);
  switch (info.nd) {
    case 1:
      {
        bob::sp::DCT1DKiss op(info.shape[0]);
        blitz::Array<double,1> res_ = res.bz<double,1>();
        op(ar.bz<double,1>(), res_);
      }
      break;
    case 2:
      {
        bob::sp::DCT2DKiss op(info.shape[0], info.shape[1]);
        blitz::Array<double,2> res_ = res.bz<double,2>();
        op(ar.bz<double,2>(), res_);
      }
      break;
    default:
      PYTHON_ERROR(TypeError, "DCT operation only supports 1 or 2D double input arrays - you provided an array of dimensionality '" SIZE_T_FMT "'.", info.nd);
  }
  return res.self();
}

static object script_idct(bob::python::const_ndarray ar) {
  const bob::core::array::typeinfo& info = ar.type();
  bob::python::ndarray res(info);
  switch (info.nd) {
    case 1:
      {
        bob::sp::IDCT1DKiss op(info.shape[0]);
        blitz::Array<double,1> res_ = res.bz<double,1>();
        op(ar.bz<double,1>(), res_);
      }
      break;
    case 2:
      {
        bob::sp::IDCT2DKiss op(info.shape[0], info.shape[1]);
        blitz::Array<double,2> res_ = res.bz<double,2>();
        op(ar.bz<double,2>(), res_);
      }
      break;
    default:
      PYTHON_ERROR(TypeError, "iDCT operation only supports 1 or 2D double input arrays - you provided an array of dimensionality '" SIZE_T_FMT "'.", info.nd);
  }
  return res.self();
}


void bind_sp_dct_kiss()
{
  // (Fast) Discrete Cosine Transform
  class_<bob::sp::DCT1DKissAbstract, boost::noncopyable>("DCT1DKissAbstract", "Abstract class for DCT1DKiss", no_init)
      .add_property("length", &bob::sp::DCT1DKiss::getLength, &bob::sp::DCT1DKiss::setLength, "Length of the array to process.")
    ;

  class_<bob::sp::DCT1DKiss, boost::shared_ptr<bob::sp::DCT1DKiss>, bases<bob::sp::DCT1DKissAbstract> >("DCT1DKiss", DCT1DKiss_DOC, init<const size_t>((arg("self"), arg("length"))))
      .def(init<bob::sp::DCT1DKiss&>((arg("self"), arg("other"))))
      .def(self == self)
      .def(self != self)
      .def("__call__", &py_dct1d_c, (arg("self"), arg("input"), arg("output")), "Compute the DCT of the input 1D array/signal. The output should have the expected size and type (numpy.float64).")
      .def("__call__", &py_dct1d_p, (arg("self"), arg("input")), "Compute the DCT of the input 1D array/signal. The output is allocated and returned.")
    ;

  class_<bob::sp::IDCT1DKiss, boost::shared_ptr<bob::sp::IDCT1DKiss>, bases<bob::sp::DCT1DKissAbstract> >("IDCT1DKiss", IDCT1DKiss_DOC, init<const size_t>((arg("self"), arg("length"))))
      .def(init<bob::sp::IDCT1DKiss&>((arg("self"), arg("other"))))
      .def(self == self)
      .def(self != self)
      .def("__call__", &py_idct1d_c, (arg("self"), arg("input"), arg("output")), "Compute the inverse DCT of the input 1D array/signal. The output should have the expected size and type (numpy.float64).")
      .def("__call__", &py_idct1d_p, (arg("self"), arg("input")), "Compute the inverse DCT of the input 1D array/signal. The output is allocated and returned.")
    ;

  class_<bob::sp::DCT2DKissAbstract, boost::noncopyable>("DCT2DKissAbstract", "Abstract class for DCT2DKiss", no_init)
      .add_property("height", &bob::sp::DCT2DKiss::getHeight, &bob::sp::DCT2DKiss::setHeight, "Height of the array to process.")
      .add_property("width", &bob::sp::DCT2DKiss::getWidth, &bob::sp::DCT2DKiss::setWidth, "Width of the array to process.")
    ;

  class_<bob::sp::DCT2DKiss, boost::shared_ptr<bob::sp::DCT2DKiss>, bases<bob::sp::DCT2DKissAbstract> >("DCT2DKiss", DCT2DKiss_DOC, init<const size_t, const size_t>((arg("self"), arg("height"), arg("width"))))
      .def(init<bob::sp::DCT2DKiss&>((arg("self"), arg("other"))))
      .def(self == self)
      .def(self != self)
      .def("__call__", &py_dct2d_c, (arg("self"), arg("input"), arg("output")), "Compute the DCT of the input 2D array/signal. The output should have the expected size and type (numpy.float64).")
      .def("__call__", &py_dct2d_p, (arg("self"), arg("input")), "Compute the DCT of the input 2D array/signal. The output is allocated and returned.")
    ;

  class_<bob::sp::IDCT2DKiss, boost::shared_ptr<bob::sp::IDCT2DKiss>, bases<bob::sp::DCT2DKissAbstract> >("IDCT2DKiss", IDCT2DKiss_DOC, init<const size_t, const size_t>((arg("self"), arg("height"), arg("width"))))
      .def(init<bob::sp::IDCT2DKiss&>((arg("self"), arg("other"))))
      .def(self == self)
      .def(self != self)
      .def("__call__", &py_idct2d_c, (arg("self"), arg("input"), arg("output")), "Compute the inverse DCT of the input 2D array/signal. The output should have the expected size and type (numpy.float64).")
      .def("__call__", &py_idct2d_p, (arg("self"), arg("input")), "Compute the inverse DCT of the input 2D array/signal. The output is allocated and returned.")
    ;

  // dct function-like
  def("dct_kiss", &script_dct, (arg("array")), DCT_DOC);
  def("idct_kiss", &script_idct, (arg("array")), IDCT_DOC);
}
