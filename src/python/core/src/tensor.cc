/**
 * @file python/core/src/tensor.cc
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 *
 * @brief Binds the Tensor object type into python
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

#include <boost/python.hpp>

#include "core/Tensor.h"

using namespace boost::python;

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(resize_overloads, resize, 1, 4)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(get_overloads, get, 1, 4)

static int ts_get_size(const bob::TensorSize& t, int i) {
  if (i < t.n_dimensions) return t.size[i];
  return 0;
}

static void ts_set_size(bob::TensorSize& t, int i, int v) {
  if (i < t.n_dimensions) t.size[i] = v;
}

static int tr_get_size(const bob::TensorRegion& t, int i) {
  if (i < t.n_dimensions) return t.size[i];
  return 0;
}

static void tr_set_size(bob::TensorRegion& t, int i, long v) {
  if (i < t.n_dimensions) t.size[i] = v;
}

static int tr_get_pos(const bob::TensorRegion& t, int i) {
  if (i < t.n_dimensions) return t.pos[i];
  return 0;
}

static void tr_set_pos(bob::TensorRegion& t, int i, long v) {
  if (i < t.n_dimensions) t.pos[i] = v;
}

void bind_core_tensor()
{
  class_<bob::TensorSize>("TensorSize", "Structures of this type represent multi-dimensional tensor sizes.", init<>("Default constructor"))
    .def(init<int>(arg("dim0"), "Initializes structure with a single dimension"))
    .def(init<int, int>((arg("dim0"), arg("dim1")), "Initializes structure with 2 dimensions"))
    .def(init<int, int, int>((arg("dim0"), arg("dim1"), arg("dim2")), "Initializes structure with 3 dimensions"))
    .def(init<int, int, int, int>((arg("dim0"), arg("dim1"), arg("dim2"), arg("dim3")), "Initializes structure with 4 dimensions"))
    .def_readwrite("n_dimensions", &bob::TensorSize::n_dimensions)
    .add_property("size", &ts_get_size, &ts_set_size)
    ;

  class_<bob::TensorRegion>("TensorRegion", "A tensor region represents a slice of the tensor data", init<>("Default constructor"))
    .def(init<long, long>((arg("d0_start"), arg("d0_size")), "One-dimesion initialization"))
    .def(init<long, long, long, long>((arg("d0_start"), arg("d0_size"), arg("d1_start"), arg("d1_size")), "Two-dimesion initialization"))
    .def(init<long, long, long, long, long, long>((arg("d0_start"), arg("d0_size"), arg("d1_start"), arg("d1_size"), arg("d2_start"), arg("d2_size")), "Three-dimesion initialization"))
    .def(init<long, long, long, long, long, long, long, long>((arg("d0_start"), arg("d0_size"), arg("d1_start"), arg("d1_size"), arg("d2_start"), arg("d2_size"), arg("d3_start"), arg("d3_size")), "Four-dimesion initialization"))
    .def_readwrite("n_dimensions", &bob::TensorRegion::n_dimensions)
    .add_property("size", &tr_get_size, &tr_set_size)
    .add_property("pos", &tr_get_pos, &tr_set_pos)
    ;

  enum_<bob::Tensor::Type>("Type")
    .value("Char", bob::Tensor::Char)
    .value("Short", bob::Tensor::Short)
    .value("Int", bob::Tensor::Int)
    .value("Long", bob::Tensor::Long)
    .value("Float", bob::Tensor::Float)
    .value("Double", bob::Tensor::Double)
    ;
  class_<bob::Tensor, boost::noncopyable>("Tensor", "The Tensor class is the base class for all Tensor types in bob", no_init)
    .def("getDatatype", &bob::Tensor::getDatatype, arg("self"), "Returns the type of data this tensor holds")
    .def("nDimension", &bob::Tensor::nDimension, arg("self"), "Returns the number of dimensions of the tensor")
    .def("size", &bob::Tensor::size, (arg("self"), arg("dimension")), "Returns the size of the tensor along a dimension")
    .def("sizeAll", &bob::Tensor::sizeAll, arg("self"), "Returns the total number of elements in this tensor")
    .def("setTensor", &bob::Tensor::setTensor, with_custodian_and_ward<1, 2>(), (arg("self"), arg("other")), "Sets this tensor with the values of the other tensor without copying (this will create a reference)")
    .def("copy", &bob::Tensor::copy, (arg("self"), arg("other")), "Sets this tensor with the values of the other tensor by copying element-by-element")
    .def("transpose", &bob::Tensor::transpose, with_custodian_and_ward<1, 2>(), (arg("self"), arg("source"), arg("dimension_1"), arg("dimension_2")), "Transposes 2 dimensions of this tensor and returns a new tensor")
    .def("narrow", &bob::Tensor::narrow, (arg("self"), arg("source"), arg("dimension"), arg("first_index"), arg("size")), "Slices a tensor and returns a new tensor containing the sliced part")
    .def("select", (void (bob::Tensor::*)(const bob::Tensor*, int, long))&bob::Tensor::select, with_custodian_and_ward<1, 2>(), (arg("self"), arg("source"), arg("dimension"), arg("slice_index")), "Selects a tensor along a certain dimension at the given slice index")
    .def("select", (bob::Tensor* (bob::Tensor::*)(int, long) const)&bob::Tensor::select, with_custodian_and_ward_postcall<0, 1, return_value_policy<manage_new_object> >(), (arg("self"), arg("dimension"), arg("slice_index")), "Returns a new tensor that is selected from mysel according to the dimension and slice_index settings")
    .def("typeSize", &bob::Tensor::typeSize, arg("self"), "Returns the size of one of my elements")
    .def("isReference", &bob::Tensor::isReference, arg("self"), "Tells if this tensor contains data or is just a reference to another tensor")
    .def("resize", (void (bob::Tensor::*)(long, long, long, long))&bob::Tensor::resize, resize_overloads((arg("dimension0"), arg("dimension1"), arg("dimension2"), arg("dimension3")), "Resizes the tensor"))
    ;

  //the several specialization for the Tensor class
  class_<bob::CharTensor, bases<bob::Tensor> >("CharTensor",
      init<optional<long, long, long, long> >((arg("dimension0"), arg("dimension1"), arg("dimension2"), arg("dimension3")), "A tensor with char values"))
    .def("get", (char (bob::CharTensor::*)(long, long, long, long) const)&bob::CharTensor::get, get_overloads((arg("dimension0"), arg("dimension1"), arg("dimension2"), arg("dimension3")), "Returns a specific value from the tensor"))
    .def("set", (void (bob::CharTensor::*)(long, char))&bob::CharTensor::set, (arg("self"), arg("index0"), arg("value")), "Sets a value in a one-dimensional char tensor")
    .def("set", (void (bob::CharTensor::*)(long, long, char))&bob::CharTensor::set, (arg("self"), arg("index0"), arg("index1"), arg("value")), "Sets a value in a two-dimensional char tensor")
    .def("set", (void (bob::CharTensor::*)(long, long, long, char))&bob::CharTensor::set, (arg("self"), arg("index0"), arg("index1"), arg("index2"), arg("value")), "Sets a value in a three-dimensional char tensor")
    .def("set", (void (bob::CharTensor::*)(long, long, long, long, char))&bob::CharTensor::set, (arg("self"), arg("index0"), arg("index1"), arg("index2"), arg("index3"), arg("value")), "Sets a value in a four-dimensional char tensor")
    .def("sum", &bob::CharTensor::sum, (arg("self")), "Returns the sume of all elements")
    .def("fill", &bob::CharTensor::fill, (arg("self"), arg("value")), "Sets all elements of the tensor to the given value")
    ;

  class_<bob::ShortTensor, bases<bob::Tensor> >("ShortTensor",
      init<optional<long, long, long, long> >((arg("dimension0"), arg("dimension1"), arg("dimension2"), arg("dimension3")), "A tensor with short integer values"))
    .def("get", (short (bob::ShortTensor::*)(long, long, long, long) const)&bob::ShortTensor::get, get_overloads((arg("dimension0"), arg("dimension1"), arg("dimension2"), arg("dimension3")), "Returns a specific value from the tensor"))
    .def("set", (void (bob::ShortTensor::*)(long, short))&bob::ShortTensor::set, (arg("self"), arg("index0"), arg("value")), "Sets a value in a one-dimensional short tensor")
    .def("set", (void (bob::ShortTensor::*)(long, long, short))&bob::ShortTensor::set, (arg("self"), arg("index0"), arg("index1"), arg("value")), "Sets a value in a two-dimensional short tensor")
    .def("set", (void (bob::ShortTensor::*)(long, long, long, short))&bob::ShortTensor::set, (arg("self"), arg("index0"), arg("index1"), arg("index2"), arg("value")), "Sets a value in a three-dimensional short tensor")
    .def("set", (void (bob::ShortTensor::*)(long, long, long, long, short))&bob::ShortTensor::set, (arg("self"), arg("index0"), arg("index1"), arg("index2"), arg("index3"), arg("value")), "Sets a value in a four-dimensional short tensor")
    .def("sum", &bob::ShortTensor::sum, (arg("self")), "Returns the sume of all elements")
    .def("fill", &bob::ShortTensor::fill, (arg("self"), arg("value")), "Sets all elements of the tensor to the given value")
    ;

  class_<bob::IntTensor, bases<bob::Tensor> >("IntTensor",
      init<optional<long, long, long, long> >((arg("dimension0"), arg("dimension1"), arg("dimension2"), arg("dimension3")), "A tensor with integer values"))
    .def("get", (int (bob::IntTensor::*)(long, long, long, long) const)&bob::IntTensor::get, get_overloads((arg("dimension0"), arg("dimension1"), arg("dimension2"), arg("dimension3")), "Returns a specific value from the tensor"))
    .def("set", (void (bob::IntTensor::*)(long, int))&bob::IntTensor::set, (arg("self"), arg("index0"), arg("value")), "Sets a value in a one-dimensional int tensor")
    .def("set", (void (bob::IntTensor::*)(long, long, int))&bob::IntTensor::set, (arg("self"), arg("index0"), arg("index1"), arg("value")), "Sets a value in a two-dimensional int tensor")
    .def("set", (void (bob::IntTensor::*)(long, long, long, int))&bob::IntTensor::set, (arg("self"), arg("index0"), arg("index1"), arg("index2"), arg("value")), "Sets a value in a three-dimensional int tensor")
    .def("set", (void (bob::IntTensor::*)(long, long, long, long, int))&bob::IntTensor::set, (arg("self"), arg("index0"), arg("index1"), arg("index2"), arg("index3"), arg("value")), "Sets a value in a four-dimensional int tensor")
    .def("sum", &bob::IntTensor::sum, (arg("self")), "Returns the sume of all elements")
    .def("fill", &bob::IntTensor::fill, (arg("self"), arg("value")), "Sets all elements of the tensor to the given value")
    ;

  class_<bob::LongTensor, bases<bob::Tensor> >("LongTensor",
      init<optional<long, long, long, long> >((arg("dimension0"), arg("dimension1"), arg("dimension2"), arg("dimension3")), "A tensor with long integer values"))
    .def("get", (long (bob::LongTensor::*)(long, long, long, long) const)&bob::LongTensor::get, get_overloads((arg("dimension0"), arg("dimension1"), arg("dimension2"), arg("dimension3")), "Returns a specific value from the tensor"))
    .def("set", (void (bob::LongTensor::*)(long, long))&bob::LongTensor::set, (arg("self"), arg("index0"), arg("value")), "Sets a value in a one-dimensional long tensor")
    .def("set", (void (bob::LongTensor::*)(long, long, long))&bob::LongTensor::set, (arg("self"), arg("index0"), arg("index1"), arg("value")), "Sets a value in a two-dimensional long tensor")
    .def("set", (void (bob::LongTensor::*)(long, long, long, long))&bob::LongTensor::set, (arg("self"), arg("index0"), arg("index1"), arg("index2"), arg("value")), "Sets a value in a three-dimensional long tensor")
    .def("set", (void (bob::LongTensor::*)(long, long, long, long, long))&bob::LongTensor::set, (arg("self"), arg("index0"), arg("index1"), arg("index2"), arg("index3"), arg("value")), "Sets a value in a four-dimensional long tensor")
    .def("sum", &bob::LongTensor::sum, (arg("self")), "Returns the sume of all elements")
    .def("fill", &bob::LongTensor::fill, (arg("self"), arg("value")), "Sets all elements of the tensor to the given value")
    ;

  class_<bob::FloatTensor, bases<bob::Tensor> >("FloatTensor",
      init<optional<long, long, long, long> >((arg("dimension0"), arg("dimension1"), arg("dimension2"), arg("dimension3")), "A tensor with float-point values"))
    .def("get", (float (bob::FloatTensor::*)(long, long, long, long) const)&bob::FloatTensor::get, get_overloads((arg("dimension0"), arg("dimension1"), arg("dimension2"), arg("dimension3")), "Returns a specific value from the tensor"))
    .def("set", (void (bob::FloatTensor::*)(long, float))&bob::FloatTensor::set, (arg("self"), arg("index0"), arg("value")), "Sets a value in a one-dimensional float tensor")
    .def("set", (void (bob::FloatTensor::*)(long, long, float))&bob::FloatTensor::set, (arg("self"), arg("index0"), arg("index1"), arg("value")), "Sets a value in a two-dimensional float tensor")
    .def("set", (void (bob::FloatTensor::*)(long, long, long, float))&bob::FloatTensor::set, (arg("self"), arg("index0"), arg("index1"), arg("index2"), arg("value")), "Sets a value in a three-dimensional float tensor")
    .def("set", (void (bob::FloatTensor::*)(long, long, long, long, float))&bob::FloatTensor::set, (arg("self"), arg("index0"), arg("index1"), arg("index2"), arg("index3"), arg("value")), "Sets a value in a four-dimensional float tensor")
    .def("sum", &bob::FloatTensor::sum, (arg("self")), "Returns the sume of all elements")
    .def("fill", &bob::FloatTensor::fill, (arg("self"), arg("value")), "Sets all elements of the tensor to the given value")
    ;

  class_<bob::DoubleTensor, bases<bob::Tensor> >("DoubleTensor",
      init<optional<long, long, long, long> >((arg("dimension0"), arg("dimension1"), arg("dimension2"), arg("dimension3")), "A tensor with double-precision values values"))
    .def("get", (double (bob::DoubleTensor::*)(long, long, long, long) const)&bob::DoubleTensor::get, get_overloads((arg("dimension0"), arg("dimension1"), arg("dimension2"), arg("dimension3")), "Returns a specific value from the tensor"))
    .def("set", (void (bob::DoubleTensor::*)(long, double))&bob::DoubleTensor::set, (arg("self"), arg("index0"), arg("value")), "Sets a value in a one-dimensional double tensor")
    .def("set", (void (bob::DoubleTensor::*)(long, long, double))&bob::DoubleTensor::set, (arg("self"), arg("index0"), arg("index1"), arg("value")), "Sets a value in a two-dimensional double tensor")
    .def("set", (void (bob::DoubleTensor::*)(long, long, long, double))&bob::DoubleTensor::set, (arg("self"), arg("index0"), arg("index1"), arg("index2"), arg("value")), "Sets a value in a three-dimensional double tensor")
    .def("set", (void (bob::DoubleTensor::*)(long, long, long, long, double))&bob::DoubleTensor::set, (arg("self"), arg("index0"), arg("index1"), arg("index2"), arg("index3"), arg("value")), "Sets a value in a four-dimensional double tensor")
    .def("sum", &bob::DoubleTensor::sum, (arg("self")), "Returns the sume of all elements")
    .def("fill", &bob::DoubleTensor::fill, (arg("self"), arg("value")), "Sets all elements of the tensor to the given value")
    ;
}
