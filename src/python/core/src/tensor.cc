/**
 * @file src/core/python/src/tensor.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a>
 *
 * @brief Binds the Tensor object type into python
 */

#include <boost/python.hpp>

#include "core/Tensor.h"

using namespace boost::python;

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(resize_overloads, resize, 1, 4)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(get_overloads, get, 1, 4)

static int ts_get_size(const Torch::TensorSize& t, int i) {
  if (i < t.n_dimensions) return t.size[i];
  return 0;
}

static void ts_set_size(Torch::TensorSize& t, int i, int v) {
  if (i < t.n_dimensions) t.size[i] = v;
}

static int tr_get_size(const Torch::TensorRegion& t, int i) {
  if (i < t.n_dimensions) return t.size[i];
  return 0;
}

static void tr_set_size(Torch::TensorRegion& t, int i, long v) {
  if (i < t.n_dimensions) t.size[i] = v;
}

static int tr_get_pos(const Torch::TensorRegion& t, int i) {
  if (i < t.n_dimensions) return t.pos[i];
  return 0;
}

static void tr_set_pos(Torch::TensorRegion& t, int i, long v) {
  if (i < t.n_dimensions) t.pos[i] = v;
}

void bind_core_tensor()
{
  class_<Torch::TensorSize>("TensorSize", "Structures of this type represent multi-dimensional tensor sizes.", init<>("Default constructor"))
    .def(init<int>(arg("dim0"), "Initializes structure with a single dimension"))
    .def(init<int, int>((arg("dim0"), arg("dim1")), "Initializes structure with 2 dimensions"))
    .def(init<int, int, int>((arg("dim0"), arg("dim1"), arg("dim2")), "Initializes structure with 3 dimensions"))
    .def(init<int, int, int, int>((arg("dim0"), arg("dim1"), arg("dim2"), arg("dim3")), "Initializes structure with 4 dimensions"))
    .def_readwrite("n_dimensions", &Torch::TensorSize::n_dimensions)
    .add_property("size", &ts_get_size, &ts_set_size)
    ;

  class_<Torch::TensorRegion>("TensorRegion", "A tensor region represents a slice of the tensor data", init<>("Default constructor"))
    .def(init<long, long>((arg("d0_start"), arg("d0_size")), "One-dimesion initialization"))
    .def(init<long, long, long, long>((arg("d0_start"), arg("d0_size"), arg("d1_start"), arg("d1_size")), "Two-dimesion initialization"))
    .def(init<long, long, long, long, long, long>((arg("d0_start"), arg("d0_size"), arg("d1_start"), arg("d1_size"), arg("d2_start"), arg("d2_size")), "Three-dimesion initialization"))
    .def(init<long, long, long, long, long, long, long, long>((arg("d0_start"), arg("d0_size"), arg("d1_start"), arg("d1_size"), arg("d2_start"), arg("d2_size"), arg("d3_start"), arg("d3_size")), "Four-dimesion initialization"))
    .def_readwrite("n_dimensions", &Torch::TensorRegion::n_dimensions)
    .add_property("size", &tr_get_size, &tr_set_size)
    .add_property("pos", &tr_get_pos, &tr_set_pos)
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
    .def("resize", (void (Torch::Tensor::*)(long, long, long, long))&Torch::Tensor::resize, resize_overloads((arg("dimension0"), arg("dimension1"), arg("dimension2"), arg("dimension3")), "Resizes the tensor"))
    ;

  //the several specialization for the Tensor class
  class_<Torch::CharTensor, bases<Torch::Tensor> >("CharTensor",
      init<optional<long, long, long, long> >((arg("dimension0"), arg("dimension1"), arg("dimension2"), arg("dimension3")), "A tensor with char values"))
    .def("get", (char (Torch::CharTensor::*)(long, long, long, long) const)&Torch::CharTensor::get, get_overloads((arg("dimension0"), arg("dimension1"), arg("dimension2"), arg("dimension3")), "Returns a specific value from the tensor"))
    .def("set", (void (Torch::CharTensor::*)(long, char))&Torch::CharTensor::set, (arg("self"), arg("index0"), arg("value")), "Sets a value in a one-dimensional char tensor")
    .def("set", (void (Torch::CharTensor::*)(long, long, char))&Torch::CharTensor::set, (arg("self"), arg("index0"), arg("index1"), arg("value")), "Sets a value in a two-dimensional char tensor")
    .def("set", (void (Torch::CharTensor::*)(long, long, long, char))&Torch::CharTensor::set, (arg("self"), arg("index0"), arg("index1"), arg("index2"), arg("value")), "Sets a value in a three-dimensional char tensor")
    .def("set", (void (Torch::CharTensor::*)(long, long, long, long, char))&Torch::CharTensor::set, (arg("self"), arg("index0"), arg("index1"), arg("index2"), arg("index3"), arg("value")), "Sets a value in a four-dimensional char tensor")
    .def("sum", &Torch::CharTensor::sum, (arg("self")), "Returns the sume of all elements")
    .def("fill", &Torch::CharTensor::fill, (arg("self"), arg("value")), "Sets all elements of the tensor to the given value")
    ;

  class_<Torch::ShortTensor, bases<Torch::Tensor> >("ShortTensor",
      init<optional<long, long, long, long> >((arg("dimension0"), arg("dimension1"), arg("dimension2"), arg("dimension3")), "A tensor with short integer values"))
    .def("get", (short (Torch::ShortTensor::*)(long, long, long, long) const)&Torch::ShortTensor::get, get_overloads((arg("dimension0"), arg("dimension1"), arg("dimension2"), arg("dimension3")), "Returns a specific value from the tensor"))
    .def("set", (void (Torch::ShortTensor::*)(long, short))&Torch::ShortTensor::set, (arg("self"), arg("index0"), arg("value")), "Sets a value in a one-dimensional short tensor")
    .def("set", (void (Torch::ShortTensor::*)(long, long, short))&Torch::ShortTensor::set, (arg("self"), arg("index0"), arg("index1"), arg("value")), "Sets a value in a two-dimensional short tensor")
    .def("set", (void (Torch::ShortTensor::*)(long, long, long, short))&Torch::ShortTensor::set, (arg("self"), arg("index0"), arg("index1"), arg("index2"), arg("value")), "Sets a value in a three-dimensional short tensor")
    .def("set", (void (Torch::ShortTensor::*)(long, long, long, long, short))&Torch::ShortTensor::set, (arg("self"), arg("index0"), arg("index1"), arg("index2"), arg("index3"), arg("value")), "Sets a value in a four-dimensional short tensor")
    .def("sum", &Torch::ShortTensor::sum, (arg("self")), "Returns the sume of all elements")
    .def("fill", &Torch::ShortTensor::fill, (arg("self"), arg("value")), "Sets all elements of the tensor to the given value")
    ;

  class_<Torch::IntTensor, bases<Torch::Tensor> >("IntTensor",
      init<optional<long, long, long, long> >((arg("dimension0"), arg("dimension1"), arg("dimension2"), arg("dimension3")), "A tensor with integer values"))
    .def("get", (int (Torch::IntTensor::*)(long, long, long, long) const)&Torch::IntTensor::get, get_overloads((arg("dimension0"), arg("dimension1"), arg("dimension2"), arg("dimension3")), "Returns a specific value from the tensor"))
    .def("set", (void (Torch::IntTensor::*)(long, int))&Torch::IntTensor::set, (arg("self"), arg("index0"), arg("value")), "Sets a value in a one-dimensional int tensor")
    .def("set", (void (Torch::IntTensor::*)(long, long, int))&Torch::IntTensor::set, (arg("self"), arg("index0"), arg("index1"), arg("value")), "Sets a value in a two-dimensional int tensor")
    .def("set", (void (Torch::IntTensor::*)(long, long, long, int))&Torch::IntTensor::set, (arg("self"), arg("index0"), arg("index1"), arg("index2"), arg("value")), "Sets a value in a three-dimensional int tensor")
    .def("set", (void (Torch::IntTensor::*)(long, long, long, long, int))&Torch::IntTensor::set, (arg("self"), arg("index0"), arg("index1"), arg("index2"), arg("index3"), arg("value")), "Sets a value in a four-dimensional int tensor")
    .def("sum", &Torch::IntTensor::sum, (arg("self")), "Returns the sume of all elements")
    .def("fill", &Torch::IntTensor::fill, (arg("self"), arg("value")), "Sets all elements of the tensor to the given value")
    ;

  class_<Torch::LongTensor, bases<Torch::Tensor> >("LongTensor",
      init<optional<long, long, long, long> >((arg("dimension0"), arg("dimension1"), arg("dimension2"), arg("dimension3")), "A tensor with long integer values"))
    .def("get", (long (Torch::LongTensor::*)(long, long, long, long) const)&Torch::LongTensor::get, get_overloads((arg("dimension0"), arg("dimension1"), arg("dimension2"), arg("dimension3")), "Returns a specific value from the tensor"))
    .def("set", (void (Torch::LongTensor::*)(long, long))&Torch::LongTensor::set, (arg("self"), arg("index0"), arg("value")), "Sets a value in a one-dimensional long tensor")
    .def("set", (void (Torch::LongTensor::*)(long, long, long))&Torch::LongTensor::set, (arg("self"), arg("index0"), arg("index1"), arg("value")), "Sets a value in a two-dimensional long tensor")
    .def("set", (void (Torch::LongTensor::*)(long, long, long, long))&Torch::LongTensor::set, (arg("self"), arg("index0"), arg("index1"), arg("index2"), arg("value")), "Sets a value in a three-dimensional long tensor")
    .def("set", (void (Torch::LongTensor::*)(long, long, long, long, long))&Torch::LongTensor::set, (arg("self"), arg("index0"), arg("index1"), arg("index2"), arg("index3"), arg("value")), "Sets a value in a four-dimensional long tensor")
    .def("sum", &Torch::LongTensor::sum, (arg("self")), "Returns the sume of all elements")
    .def("fill", &Torch::LongTensor::fill, (arg("self"), arg("value")), "Sets all elements of the tensor to the given value")
    ;

  class_<Torch::FloatTensor, bases<Torch::Tensor> >("FloatTensor",
      init<optional<long, long, long, long> >((arg("dimension0"), arg("dimension1"), arg("dimension2"), arg("dimension3")), "A tensor with float-point values"))
    .def("get", (float (Torch::FloatTensor::*)(long, long, long, long) const)&Torch::FloatTensor::get, get_overloads((arg("dimension0"), arg("dimension1"), arg("dimension2"), arg("dimension3")), "Returns a specific value from the tensor"))
    .def("set", (void (Torch::FloatTensor::*)(long, float))&Torch::FloatTensor::set, (arg("self"), arg("index0"), arg("value")), "Sets a value in a one-dimensional float tensor")
    .def("set", (void (Torch::FloatTensor::*)(long, long, float))&Torch::FloatTensor::set, (arg("self"), arg("index0"), arg("index1"), arg("value")), "Sets a value in a two-dimensional float tensor")
    .def("set", (void (Torch::FloatTensor::*)(long, long, long, float))&Torch::FloatTensor::set, (arg("self"), arg("index0"), arg("index1"), arg("index2"), arg("value")), "Sets a value in a three-dimensional float tensor")
    .def("set", (void (Torch::FloatTensor::*)(long, long, long, long, float))&Torch::FloatTensor::set, (arg("self"), arg("index0"), arg("index1"), arg("index2"), arg("index3"), arg("value")), "Sets a value in a four-dimensional float tensor")
    .def("sum", &Torch::FloatTensor::sum, (arg("self")), "Returns the sume of all elements")
    .def("fill", &Torch::FloatTensor::fill, (arg("self"), arg("value")), "Sets all elements of the tensor to the given value")
    ;

  class_<Torch::DoubleTensor, bases<Torch::Tensor> >("DoubleTensor",
      init<optional<long, long, long, long> >((arg("dimension0"), arg("dimension1"), arg("dimension2"), arg("dimension3")), "A tensor with double-precision values values"))
    .def("get", (double (Torch::DoubleTensor::*)(long, long, long, long) const)&Torch::DoubleTensor::get, get_overloads((arg("dimension0"), arg("dimension1"), arg("dimension2"), arg("dimension3")), "Returns a specific value from the tensor"))
    .def("set", (void (Torch::DoubleTensor::*)(long, double))&Torch::DoubleTensor::set, (arg("self"), arg("index0"), arg("value")), "Sets a value in a one-dimensional double tensor")
    .def("set", (void (Torch::DoubleTensor::*)(long, long, double))&Torch::DoubleTensor::set, (arg("self"), arg("index0"), arg("index1"), arg("value")), "Sets a value in a two-dimensional double tensor")
    .def("set", (void (Torch::DoubleTensor::*)(long, long, long, double))&Torch::DoubleTensor::set, (arg("self"), arg("index0"), arg("index1"), arg("index2"), arg("value")), "Sets a value in a three-dimensional double tensor")
    .def("set", (void (Torch::DoubleTensor::*)(long, long, long, long, double))&Torch::DoubleTensor::set, (arg("self"), arg("index0"), arg("index1"), arg("index2"), arg("index3"), arg("value")), "Sets a value in a four-dimensional double tensor")
    .def("sum", &Torch::DoubleTensor::sum, (arg("self")), "Returns the sume of all elements")
    .def("fill", &Torch::DoubleTensor::fill, (arg("self"), arg("value")), "Sets all elements of the tensor to the given value")
    ;
}
