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

static int ts_get_size(const Torch::TensorSize& t, unsigned int i) {
  if (i < t.n_dimensions) return t.size[i];
  return 0;
}

static void ts_set_size(Torch::TensorSize& t, unsigned int i, int v) {
  if (i < t.n_dimensions) t.size[i] = v;
}

static int tr_get_size(const Torch::TensorRegion& t, unsigned int i) {
  if (i < t.n_dimensions) return t.size[i];
  return 0;
}

static int tr_set_size(Torch::TensorRegion& t, unsigned int i, long v) {
  if (i < t.n_dimensions) t.size[i] = v;
}

static int tr_get_pos(const Torch::TensorRegion& t, unsigned int i) {
  if (i < t.n_dimensions) return t.pos[i];
  return 0;
}

static int tr_set_pos(Torch::TensorRegion& t, unsigned int i, long v) {
  if (i < t.n_dimensions) t.pos[i] = v;
}

void bind_core_tensor()
{
  class_<Torch::TensorSize>("TensorSize", init<>("Structures of this type represent multi-dimensional tensor sizes."))
    .def(init<int>(arg("dim0"), "Initializes structure with a single dimension"))
    .def(init<int, int>((arg("dim0"), arg("dim1")), "Initializes structure with 2 dimensions"))
    .def(init<int, int, int>((arg("dim0"), arg("dim1"), arg("dim2")), "Initializes structure with 3 dimensions"))
    .def(init<int, int, int, int>((arg("dim0"), arg("dim1"), arg("dim2"), arg("dim3")), "Initializes structure with 4 dimensions"))
    .def_readwrite("n_dimensions", &Torch::TensorSize::n_dimensions)
    .def("size", &ts_get_size, (arg("self"), arg("dimension")), "Returns one of the dimension values")
    .def("set_size", &ts_set_size, (arg("self"), arg("dimension")), "Sets one of the dimension values")
    ;

  class_<Torch::TensorRegion>("TensorRegion", init<>("A tensor region represents a slice of the tensor data"))
    .def(init<long, long>((arg("d0_start"), arg("d0_size")), "One-dimesion initialization"))
    .def(init<long, long, long, long>((arg("d0_start"), arg("d0_size"), arg("d1_start"), arg("d1_size")), "Two-dimesion initialization"))
    .def(init<long, long, long, long, long, long>((arg("d0_start"), arg("d0_size"), arg("d1_start"), arg("d1_size"), arg("d2_start"), arg("d2_size")), "Three-dimesion initialization"))
    .def(init<long, long, long, long, long, long, long, long>((arg("d0_start"), arg("d0_size"), arg("d1_start"), arg("d1_size"), arg("d2_start"), arg("d2_size"), arg("d3_start"), arg("d3_size")), "Four-dimesion initialization"))
    .def_readwrite("n_dimensions", &Torch::TensorRegion::n_dimensions)
    .def("size", &tr_get_size, (arg("self"), arg("dimension")), "Returns the size in one of the dimensions")
    .def("set_size", &tr_set_size, (arg("self"), arg("dimension")), "Sets the size of the slice in one of the dimensions")
    .def("pos", &tr_get_pos, (arg("self"), arg("dimension")), "Returns the start position of the slice in one of the dimensions")
    .def("set_pos", &tr_set_size, (arg("self"), arg("dimension")), "Sets the start position of the slice in one of the dimensions")
    ;

  enum_<Torch::Tensor::Type>("Type")
    .value("Char", Torch::Tensor::Char)
    .value("Short", Torch::Tensor::Short)
    .value("Int", Torch::Tensor::Int)
    .value("Long", Torch::Tensor::Long)
    .value("Float", Torch::Tensor::Float)
    .value("Double", Torch::Tensor::Double)
    ;
  class_<Torch::Tensor, boost::noncopyable>("Tensor", "The Tensor class is the base class for all Tensor types in Torch", no_init)
    .def("getDatatype", &Torch::Tensor::getDatatype, arg("self"), "Returns the type of data this tensor holds")
    .def("nDimension", &Torch::Tensor::nDimension, arg("self"), "Returns the number of dimensions of the tensor")
    .def("size", &Torch::Tensor::size, (arg("self"), arg("dimension")), "Returns the size of the tensor along a dimension")
    .def("sizeAll", &Torch::Tensor::sizeAll, arg("self"), "Returns the total number of elements in this tensor")
    .def("setTensor", &Torch::Tensor::setTensor, with_custodian_and_ward<1, 2>(), (arg("self"), arg("other")), "Sets this tensor with the values of the other tensor without copying (this will create a reference)")
    .def("copy", &Torch::Tensor::copy, (arg("self"), arg("other")), "Sets this tensor with the values of the other tensor by copying element-by-element")
    .def("transpose", &Torch::Tensor::transpose, with_custodian_and_ward<1, 2>(), (arg("self"), arg("source"), arg("dimension_1"), arg("dimension_2")), "Transposes 2 dimensions of this tensor and returns a new tensor")
    .def("narrow", &Torch::Tensor::narrow, (arg("self"), arg("source"), arg("dimension"), arg("first_index"), arg("size")), "Slices a tensor and returns a new tensor containing the sliced part")
    .def("select", (void (Torch::Tensor::*)(const Torch::Tensor*, int, long))&Torch::Tensor::select, with_custodian_and_ward<1, 2>(), (arg("self"), arg("source"), arg("dimension"), arg("slice_index")), "Selects a tensor along a certain dimension at the given slice index")
    .def("select", (Torch::Tensor* (Torch::Tensor::*)(int, long) const)&Torch::Tensor::select, with_custodian_and_ward_postcall<0, 1, return_value_policy<manage_new_object> >(), (arg("self"), arg("dimension"), arg("slice_index")), "Returns a new tensor that is selected from mysel according to the dimension and slice_index settings")
    .def("typeSize", &Torch::Tensor::typeSize, arg("self"), "Returns the size of one of my elements")
    .def("isReference", &Torch::Tensor::isReference, arg("self"), "Tells if this tensor contains data or is just a reference to another tensor")
    .def("resize", &resize_1d, (arg("self"), arg("dimension0")), "Resizes the tensor along the first dimension")
    .def("resize", &resize_2d, (arg("self"), arg("dimension0"), arg("dimension1")), "Resizes the tensor along the first two dimensions")
    .def("resize", &resize_3d, (arg("self"), arg("dimension0"), arg("dimension1"), arg("dimension2")), "Resizes the tensor along the first three dimensions")
    .def("resize", &resize_4d, (arg("self"), arg("dimension0"), arg("dimension1"), arg("dimension2"), arg("dimension3")), "Resizes the tensor along all the four dimensions")
    .def("set", &set_l_c, (arg("self"), arg("index0"), arg("value")), "Sets a value in a one-dimensional char tensor")
    .def("set", &set_ll_c, (arg("self"), arg("index0"), arg("index1"), arg("value")), "Sets a value in a two-dimensional char tensor")
    .def("set", &set_lll_c, (arg("self"), arg("index0"), arg("index1"), arg("index2"), arg("value")), "Sets a value in a three-dimensional char tensor")
    .def("set", &set_llll_c, (arg("self"), arg("index0"), arg("index1"), arg("index2"), arg("index3"), arg("value")), "Sets a value in a four-dimensional char tensor")
    .def("set", &set_l_s, (arg("self"), arg("index0"), arg("value")), "Sets a value in a one-dimensional short tensor")
    .def("set", &set_ll_s, (arg("self"), arg("index0"), arg("index1"), arg("value")), "Sets a value in a two-dimensional short tensor")
    .def("set", &set_lll_s, (arg("self"), arg("index0"), arg("index1"), arg("index2"), arg("value")), "Sets a value in a three-dimensional short tensor")
    .def("set", &set_llll_s, (arg("self"), arg("index0"), arg("index1"), arg("index2"), arg("index3"), arg("value")), "Sets a value in a four-dimensional short tensor")
    .def("set", &set_l_i, (arg("self"), arg("index0"), arg("value")), "Sets a value in a one-dimensional integer tensor")
    .def("set", &set_ll_i, (arg("self"), arg("index0"), arg("index1"), arg("value")), "Sets a value in a two-dimensional integer tensor")
    .def("set", &set_lll_i, (arg("self"), arg("index0"), arg("index1"), arg("index2"), arg("value")), "Sets a value in a three-dimensional integer tensor")
    .def("set", &set_llll_i, (arg("self"), arg("index0"), arg("index1"), arg("index2"), arg("index3"), arg("value")), "Sets a value in a four-dimensional integer tensor")
    .def("set", &set_l_l, (arg("self"), arg("index0"), arg("value")), "Sets a value in a one-dimensional long tensor")
    .def("set", &set_ll_l, (arg("self"), arg("index0"), arg("index1"), arg("value")), "Sets a value in a two-dimensional long tensor")
    .def("set", &set_lll_l, (arg("self"), arg("index0"), arg("index1"), arg("index2"), arg("value")), "Sets a value in a three-dimensional long tensor")
    .def("set", &set_llll_l, (arg("self"), arg("index0"), arg("index1"), arg("index2"), arg("index3"), arg("value")), "Sets a value in a four-dimensional long tensor")
    .def("set", &set_l_f, (arg("self"), arg("index0"), arg("value")), "Sets a value in a one-dimensional float tensor")
    .def("set", &set_ll_f, (arg("self"), arg("index0"), arg("index1"), arg("value")), "Sets a value in a two-dimensional float tensor")
    .def("set", &set_lll_f, (arg("self"), arg("index0"), arg("index1"), arg("index2"), arg("value")), "Sets a value in a three-dimensional float tensor")
    .def("set", &set_llll_f, (arg("self"), arg("index0"), arg("index1"), arg("index2"), arg("index3"), arg("value")), "Sets a value in a four-dimensional float tensor")
    .def("set", &set_l_d, (arg("self"), arg("index0"), arg("value")), "Sets a value in a one-dimensional double tensor")
    .def("set", &set_ll_d, (arg("self"), arg("index0"), arg("index1"), arg("value")), "Sets a value in a two-dimensional double tensor")
    .def("set", &set_lll_d, (arg("self"), arg("index0"), arg("index1"), arg("index2"), arg("value")), "Sets a value in a three-dimensional double tensor")
    .def("set", &set_llll_d, (arg("self"), arg("index0"), arg("index1"), arg("index2"), arg("index3"), arg("value")), "Sets a value in a four-dimensional double tensor")
    ;

    //the several specialization for the Tensor class
    class_<Torch::CharTensor, bases<Torch::Tensor> >("CharTensor", 
        init<optional<long, long, long, long> >((arg("dimension0"), arg("dimension1"), arg("dimension2"), arg("dimension3")), "A tensor with char values"));
    class_<Torch::ShortTensor, bases<Torch::Tensor> >("ShortTensor", 
        init<optional<long, long, long, long> >((arg("dimension0"), arg("dimension1"), arg("dimension2"), arg("dimension3")), "A tensor with short integer values"));
    class_<Torch::IntTensor, bases<Torch::Tensor> >("IntTensor", 
        init<optional<long, long, long, long> >((arg("dimension0"), arg("dimension1"), arg("dimension2"), arg("dimension3")), "A tensor with integer values"));
    class_<Torch::LongTensor, bases<Torch::Tensor> >("LongTensor", 
        init<optional<long, long, long, long> >((arg("dimension0"), arg("dimension1"), arg("dimension2"), arg("dimension3")), "A tensor with long integer values"));
    class_<Torch::FloatTensor, bases<Torch::Tensor> >("FloatTensor", 
        init<optional<long, long, long, long> >((arg("dimension0"), arg("dimension1"), arg("dimension2"), arg("dimension3")), "A tensor with float-point values"));
    class_<Torch::DoubleTensor, bases<Torch::Tensor> >("DoubleTensor", 
        init<optional<long, long, long, long> >((arg("dimension0"), arg("dimension1"), arg("dimension2"), arg("dimension3")), "A tensor with double-precision values values"));
}
