/**
 * @file src/core/tensor.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Binds the Tensor object type into python 
 */

#include <boost/python.hpp>

#include "core/Tensor.h"

using namespace boost::python;

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(resize_overloads, resize, 1, 4);

static void resize_1d (Torch::Tensor& o, long x) { o.resize(x); }
static void resize_2d (Torch::Tensor& o, long x, long y) { o.resize(x, y); }
static void resize_3d (Torch::Tensor& o, long x, long y, long z) { o.resize(x, y, z); }
static void resize_4d (Torch::Tensor& o, long x, long y, long z, long w) 
{ o.resize(x, y, z, w); }

static void set_l_c (Torch::Tensor& o, long x, char v) 
{ o.set(x, v); }
static void set_ll_c (Torch::Tensor& o, long x, long y, char v) 
{ o.set(x, y, v); }
static void set_lll_c (Torch::Tensor& o, long x, long y, long z,  char v) 
{ o.set(x, y, z, v); }
static void set_llll_c (Torch::Tensor& o, long x, long y, long z, long w, char v) 
{ o.set(x, y, z, w, v); }

static void set_l_s (Torch::Tensor& o, long x, short v) 
{ o.set(x, v); }
static void set_ll_s (Torch::Tensor& o, long x, long y, short v) 
{ o.set(x, y, v); }
static void set_lll_s (Torch::Tensor& o, long x, long y, long z,  short v) 
{ o.set(x, y, z, v); }
static void set_llll_s (Torch::Tensor& o, long x, long y, long z, long w, short v) 
{ o.set(x, y, z, w, v); }

static void set_l_i (Torch::Tensor& o, long x, int v) 
{ o.set(x, v); }
static void set_ll_i (Torch::Tensor& o, long x, long y, int v) 
{ o.set(x, y, v); }
static void set_lll_i (Torch::Tensor& o, long x, long y, long z,  int v) 
{ o.set(x, y, z, v); }
static void set_llll_i (Torch::Tensor& o, long x, long y, long z, long w, int v) 
{ o.set(x, y, z, w, v); }

static void set_l_l (Torch::Tensor& o, long x, long v) 
{ o.set(x, v); }
static void set_ll_l (Torch::Tensor& o, long x, long y, long v) 
{ o.set(x, y, v); }
static void set_lll_l (Torch::Tensor& o, long x, long y, long z,  long v) 
{ o.set(x, y, z, v); }
static void set_llll_l (Torch::Tensor& o, long x, long y, long z, long w, long v) 
{ o.set(x, y, z, w, v); }

static void set_l_f (Torch::Tensor& o, long x, float v) 
{ o.set(x, v); }
static void set_ll_f (Torch::Tensor& o, long x, long y, float v) 
{ o.set(x, y, v); }
static void set_lll_f (Torch::Tensor& o, long x, long y, long z,  float v) 
{ o.set(x, y, z, v); }
static void set_llll_f (Torch::Tensor& o, long x, long y, long z, long w, float v) 
{ o.set(x, y, z, w, v); }

static void set_l_d (Torch::Tensor& o, long x, double v) 
{ o.set(x, v); }
static void set_ll_d (Torch::Tensor& o, long x, long y, double v) 
{ o.set(x, y, v); }
static void set_lll_d (Torch::Tensor& o, long x, long y, long z,  double v) 
{ o.set(x, y, z, v); }
static void set_llll_d (Torch::Tensor& o, long x, long y, long z, long w, double v) 
{ o.set(x, y, z, w, v); }

void bind_core_tensor()
{
  enum_<Torch::Tensor::Type>("Type")
    .value("Char", Torch::Tensor::Char)
    .value("Short", Torch::Tensor::Short)
    .value("Int", Torch::Tensor::Int)
    .value("Long", Torch::Tensor::Long)
    .value("Float", Torch::Tensor::Float)
    .value("Double", Torch::Tensor::Double)
    ;
  class_<Torch::Tensor, boost::noncopyable>("Tensor", no_init)
    .def("getDatatype", &Torch::Tensor::getDatatype)
    .def("nDimension", &Torch::Tensor::nDimension)
    .def("size", &Torch::Tensor::size)
    .def("sizeAll", &Torch::Tensor::sizeAll)
    .def("setTensor", &Torch::Tensor::setTensor, with_custodian_and_ward<1, 2>())
    .def("copy", &Torch::Tensor::copy)
    .def("transpose", &Torch::Tensor::transpose, with_custodian_and_ward<1, 2>())
    .def("narrow", &Torch::Tensor::narrow)
    .def("select", (void (Torch::Tensor::*)(const Torch::Tensor*, int, long))&Torch::Tensor::select, with_custodian_and_ward<1, 2>())
    .def("select", (Torch::Tensor* (Torch::Tensor::*)(int, long) const)&Torch::Tensor::select, with_custodian_and_ward_postcall<0, 1, return_value_policy<manage_new_object> >())
    .def("typeSize", &Torch::Tensor::typeSize)
    .def("isReference", &Torch::Tensor::isReference)
    .def("resize", &resize_1d)
    .def("resize", &resize_2d)
    .def("resize", &resize_3d)
    .def("resize", &resize_4d)
    .def("set", &set_l_c)
    .def("set", &set_ll_c)
    .def("set", &set_lll_c)
    .def("set", &set_llll_c)
    .def("set", &set_l_s)
    .def("set", &set_ll_s)
    .def("set", &set_lll_s)
    .def("set", &set_llll_s)
    .def("set", &set_l_i)
    .def("set", &set_ll_i)
    .def("set", &set_lll_i)
    .def("set", &set_llll_i)
    .def("set", &set_l_l)
    .def("set", &set_ll_l)
    .def("set", &set_lll_l)
    .def("set", &set_llll_l)
    .def("set", &set_l_f)
    .def("set", &set_ll_f)
    .def("set", &set_lll_f)
    .def("set", &set_llll_f)
    .def("set", &set_l_d)
    .def("set", &set_ll_d)
    .def("set", &set_lll_d)
    .def("set", &set_llll_d)
    ;

    //the several specialization for the Tensor class
    class_<Torch::CharTensor, bases<Torch::Tensor> >("CharTensor", 
        init<optional<long, long, long, long> >());
    class_<Torch::ShortTensor, bases<Torch::Tensor> >("ShortTensor", 
        init<optional<long, long, long, long> >());
    class_<Torch::IntTensor, bases<Torch::Tensor> >("IntTensor", 
        init<optional<long, long, long, long> >());
    class_<Torch::LongTensor, bases<Torch::Tensor> >("LongTensor", 
        init<optional<long, long, long, long> >());
    class_<Torch::FloatTensor, bases<Torch::Tensor> >("FloatTensor", 
        init<optional<long, long, long, long> >());
    class_<Torch::DoubleTensor, bases<Torch::Tensor> >("DoubleTensor", 
        init<optional<long, long, long, long> >());
}
