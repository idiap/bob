/**
 * @file sp/python/dct_numpy.cc
 * @date Fri Nov 15 10:27:11 CET 2013
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds the (fast) Discrete Cosine Transform to python.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#include <bob/python/ndarray.h>

#include <bob/sp/DCT1DNumpy.h>
#include <bob/sp/DCT2DNumpy.h>


using namespace boost::python;

// documentation for classes
static const char* DCT1DNumpy_DOC = "Objects of this class, after configuration, can compute the direct DCT of a 1D array/signal.";
static const char* IDCT1DNumpy_DOC = "Objects of this class, after configuration, can compute the inverse DCT of a 1D array/signal.";
static const char* DCT2DNumpy_DOC = "Objects of this class, after configuration, can compute the direct DCT of a 2D array/signal.";
static const char* IDCT2DNumpy_DOC = "Objects of this class, after configuration, can compute the inverse DCT of a 2D array/signal.";

// free methods documentation
static const char* DCT_DOC = "Compute the direct DCT of a 1 or 2D array/signal of type float64.";
static const char* IDCT_DOC = "Compute the inverse DCT of a 1 or 2D array/signal of type float64.";


static void py_dct1d_c(bob::sp::DCT1DNumpy& op, bob::python::const_ndarray src,
  bob::python::ndarray dst)
{
  blitz::Array<double,1> dst_ = dst.bz<double,1>();
  op(src.bz<double,1>(), dst_);
}

static object py_dct1d_p(bob::sp::DCT1DNumpy& op, bob::python::const_ndarray src)
{
  bob::python::ndarray dst(bob::core::array::t_float64, op.getLength());
  blitz::Array<double,1> dst_ = dst.bz<double,1>();
  op(src.bz<double,1>(), dst_);
  return dst.self();
}

static void py_idct1d_c(bob::sp::IDCT1DNumpy& op, bob::python::const_ndarray src,
  bob::python::ndarray dst)
{
  blitz::Array<double,1> dst_ = dst.bz<double,1>();
  op(src.bz<double,1>(), dst_);
}

static object py_idct1d_p(bob::sp::IDCT1DNumpy& op, bob::python::const_ndarray src)
{
  bob::python::ndarray dst(bob::core::array::t_float64, op.getLength());
  blitz::Array<double,1> dst_ = dst.bz<double,1>();
  op(src.bz<double,1>(), dst_);
  return dst.self();
}


static void py_dct2d_c(bob::sp::DCT2DNumpy& op, bob::python::const_ndarray src,
  bob::python::ndarray dst)
{
  blitz::Array<double,2> dst_ = dst.bz<double,2>();
  op(src.bz<double,2>(), dst_);
}

static object py_dct2d_p(bob::sp::DCT2DNumpy& op, bob::python::const_ndarray src)
{
  bob::python::ndarray dst(bob::core::array::t_float64, op.getHeight(),
    op.getWidth());
  blitz::Array<double,2> dst_ = dst.bz<double,2>();
  op(src.bz<double,2>(), dst_);
  return dst.self();
}

static void py_idct2d_c(bob::sp::IDCT2DNumpy& op, bob::python::const_ndarray src,
  bob::python::ndarray dst)
{
  blitz::Array<double,2> dst_ = dst.bz<double,2>();
  op(src.bz<double,2>(), dst_);
}

static object py_idct2d_p(bob::sp::IDCT2DNumpy& op, bob::python::const_ndarray src)
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
        bob::sp::DCT1DNumpy op(info.shape[0]);
        blitz::Array<double,1> res_ = res.bz<double,1>();
        op(ar.bz<double,1>(), res_);
      }
      break;
    case 2:
      {
        bob::sp::DCT2DNumpy op(info.shape[0], info.shape[1]);
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
        bob::sp::IDCT1DNumpy op(info.shape[0]);
        blitz::Array<double,1> res_ = res.bz<double,1>();
        op(ar.bz<double,1>(), res_);
      }
      break;
    case 2:
      {
        bob::sp::IDCT2DNumpy op(info.shape[0], info.shape[1]);
        blitz::Array<double,2> res_ = res.bz<double,2>();
        op(ar.bz<double,2>(), res_);
      }
      break;
    default:
      PYTHON_ERROR(TypeError, "iDCT operation only supports 1 or 2D double input arrays - you provided an array of dimensionality '" SIZE_T_FMT "'.", info.nd);
  }
  return res.self();
}


void bind_sp_dct_numpy()
{
  // (Fast) Discrete Cosine Transform
  class_<bob::sp::DCT1DNumpyAbstract, boost::noncopyable>("DCT1DNumpyAbstract", "Abstract class for DCT1DNumpy", no_init)
      .add_property("length", &bob::sp::DCT1DNumpy::getLength, &bob::sp::DCT1DNumpy::setLength, "Length of the array to process.")
    ;

  class_<bob::sp::DCT1DNumpy, boost::shared_ptr<bob::sp::DCT1DNumpy>, bases<bob::sp::DCT1DNumpyAbstract> >("DCT1DNumpy", DCT1DNumpy_DOC, init<const size_t>((arg("self"), arg("length"))))
      .def(init<bob::sp::DCT1DNumpy&>((arg("self"), arg("other"))))
      .def(self == self)
      .def(self != self)
      .def("__call__", &py_dct1d_c, (arg("self"), arg("input"), arg("output")), "Compute the DCT of the input 1D array/signal. The output should have the expected size and type (numpy.float64).")
      .def("__call__", &py_dct1d_p, (arg("self"), arg("input")), "Compute the DCT of the input 1D array/signal. The output is allocated and returned.")
    ;

  class_<bob::sp::IDCT1DNumpy, boost::shared_ptr<bob::sp::IDCT1DNumpy>, bases<bob::sp::DCT1DNumpyAbstract> >("IDCT1DNumpy", IDCT1DNumpy_DOC, init<const size_t>((arg("self"), arg("length"))))
      .def(init<bob::sp::IDCT1DNumpy&>((arg("self"), arg("other"))))
      .def(self == self)
      .def(self != self)
      .def("__call__", &py_idct1d_c, (arg("self"), arg("input"), arg("output")), "Compute the inverse DCT of the input 1D array/signal. The output should have the expected size and type (numpy.float64).")
      .def("__call__", &py_idct1d_p, (arg("self"), arg("input")), "Compute the inverse DCT of the input 1D array/signal. The output is allocated and returned.")
    ;

  class_<bob::sp::DCT2DNumpyAbstract, boost::noncopyable>("DCT2DNumpyAbstract", "Abstract class for DCT2DNumpy", no_init)
      .add_property("height", &bob::sp::DCT2DNumpy::getHeight, &bob::sp::DCT2DNumpy::setHeight, "Height of the array to process.")
      .add_property("width", &bob::sp::DCT2DNumpy::getWidth, &bob::sp::DCT2DNumpy::setWidth, "Width of the array to process.")
    ;

  class_<bob::sp::DCT2DNumpy, boost::shared_ptr<bob::sp::DCT2DNumpy>, bases<bob::sp::DCT2DNumpyAbstract> >("DCT2DNumpy", DCT2DNumpy_DOC, init<const size_t, const size_t>((arg("self"), arg("height"), arg("width"))))
      .def(init<bob::sp::DCT2DNumpy&>((arg("self"), arg("other"))))
      .def(self == self)
      .def(self != self)
      .def("__call__", &py_dct2d_c, (arg("self"), arg("input"), arg("output")), "Compute the DCT of the input 2D array/signal. The output should have the expected size and type (numpy.float64).")
      .def("__call__", &py_dct2d_p, (arg("self"), arg("input")), "Compute the DCT of the input 2D array/signal. The output is allocated and returned.")
    ;

  class_<bob::sp::IDCT2DNumpy, boost::shared_ptr<bob::sp::IDCT2DNumpy>, bases<bob::sp::DCT2DNumpyAbstract> >("IDCT2DNumpy", IDCT2DNumpy_DOC, init<const size_t, const size_t>((arg("self"), arg("height"), arg("width"))))
      .def(init<bob::sp::IDCT2DNumpy&>((arg("self"), arg("other"))))
      .def(self == self)
      .def(self != self)
      .def("__call__", &py_idct2d_c, (arg("self"), arg("input"), arg("output")), "Compute the inverse DCT of the input 2D array/signal. The output should have the expected size and type (numpy.float64).")
      .def("__call__", &py_idct2d_p, (arg("self"), arg("input")), "Compute the inverse DCT of the input 2D array/signal. The output is allocated and returned.")
    ;

  // dct function-like
  def("dct_numpy", &script_dct, (arg("array")), DCT_DOC);
  def("idct_numpy", &script_idct, (arg("array")), IDCT_DOC);
}
