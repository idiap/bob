/**
 * @file src/python/core/src/array.cc
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 *
 * @brief blitz::Array<> to and from python converters for arrays
 */

#include <boost/shared_array.hpp>
#include "core/python/array.h"

template <int N> 
boost::python::class_<blitz::GeneralArrayStorage<N>,
  boost::shared_ptr<blitz::GeneralArrayStorage<N> > > bind_c_storage() {

  typedef typename blitz::GeneralArrayStorage<N> storage_type;
  boost::format class_name("c_storage_%d");
  class_name % N;
  boost::format class_doc("Storages of this type can be used to force a certain storage style in an array. Use objects of this type to force a C-storage type for a %d-D array.");
  class_doc % N;
  boost::python::class_<storage_type, boost::shared_ptr<storage_type> > 
    retval(class_name.str().c_str(), class_doc.str().c_str(), boost::python::init<>());
  return retval;
}

template <int N>
boost::python::class_<blitz::FortranArray<N>,
  boost::shared_ptr<blitz::FortranArray<N> > > bind_fortran_storage() {
  typedef typename blitz::FortranArray<N> storage_type;
  boost::format class_name("fortran_storage_%d");
  class_name % N;
  boost::format class_doc("Storages of this type can be used to force a certain storage style in an array. Use objects of this type to force a Fortran-storage type for a %d-D array.");
  class_doc % N;
  boost::python::class_<storage_type, boost::shared_ptr<storage_type> > 
    retval(class_name.str().c_str(), class_doc.str().c_str(), boost::python::init<>());
  return retval;
}

int Torch::python::check_array_limits(int index, int base, int extent) {
  const int limit = base + extent;
  index = (index<0)? index + limit : index;
  //checks final range
  if (index < base) {
    PyErr_SetString(PyExc_IndexError, "(fortran) array index out of range");
    boost::python::throw_error_already_set();
  }
  if (index >= limit) {
    PyErr_SetString(PyExc_IndexError, "array index out of range");
    boost::python::throw_error_already_set();
  }
  return index;
}

void Torch::python::check_are_slices(int size, boost::python::tuple ranges) {
  if (size != boost::python::len(ranges)) {
    boost::format s("wrong number of slices - expected %d, got %d");
    s % size % boost::python::len(ranges);
    PyErr_SetString(PyExc_TypeError, s.str().c_str());
    boost::python::throw_error_already_set();
  }
}

blitz::Range Torch::python::slice2range(boost::python::slice s, int base,
    int extent) {
  int step = 1;
  if (s.step().ptr() != Py_None) step = boost::python::extract<int>(s.step())();
  int start = 0;
  if (s.start().ptr() != Py_None) 
    start = Torch::python::check_array_limits(boost::python::extract<int>(s.start())(), base, extent); 
  int stop = extent - 1;
  if (s.stop().ptr() != Py_None) { 
    stop = Torch::python::check_array_limits(boost::python::extract<int>(s.stop())(), base, extent);
    if (step < 0) stop += 1;
    else if (stop > 0) stop -= 1;
  }
  return blitz::Range(start, stop, step); 
}

#define bind_storages(N) bind_c_storage<N>(); bind_fortran_storage<N>();

void bind_core_array() {
  boost::python::enum_<NPY_TYPES>("NPY_TYPES")
    .value("NPY_BOOL", NPY_BOOL)
    .value("NPY_BYTE", NPY_BYTE)
    .value("NPY_UBYTE", NPY_UBYTE)
    .value("NPY_SHORT", NPY_SHORT)
    .value("NPY_USHORT", NPY_USHORT)
    .value("NPY_INT", NPY_INT)
    .value("NPY_UINT", NPY_UINT)
    .value("NPY_LONG", NPY_LONG)
    .value("NPY_ULONG", NPY_ULONG)
    .value("NPY_LONGLONG", NPY_LONGLONG)
    .value("NPY_ULONGLONG", NPY_ULONGLONG)
    .value("NPY_FLOAT", NPY_FLOAT)
    .value("NPY_DOUBLE", NPY_DOUBLE)
    .value("NPY_LONGDOUBLE", NPY_LONGDOUBLE)
    .value("NPY_CFLOAT", NPY_CFLOAT)
    .value("NPY_CDOUBLE", NPY_CDOUBLE)
    .value("NPY_CLONGDOUBLE", NPY_CLONGDOUBLE)
    .value("NPY_OBJECT", NPY_OBJECT)
    .value("NPY_STRING", NPY_STRING)
    .value("NPY_UNICODE", NPY_UNICODE)
    .value("NPY_VOID", NPY_VOID)
    .value("NPY_NTYPES", NPY_NTYPES)
    .value("NPY_NOTYPE", NPY_NOTYPE)
    .value("NPY_CHAR", NPY_CHAR)
    .value("NPY_USERDEF", NPY_USERDEF)
    ;

  //some constants to make your code clearer
  boost::python::scope().attr("firstDim") = blitz::firstDim;
  boost::python::scope().attr("secondDim") = blitz::secondDim;
  boost::python::scope().attr("thirdDim") = blitz::thirdDim;
  boost::python::scope().attr("fourthDim") = blitz::fourthDim;
  boost::python::scope().attr("fifthDim") = blitz::fifthDim;
  boost::python::scope().attr("sixthDim") = blitz::sixthDim;
  boost::python::scope().attr("seventhDim") = blitz::seventhDim;
  boost::python::scope().attr("eighthDim") = blitz::eighthDim;
  boost::python::scope().attr("ninthDim") = blitz::ninthDim;
  boost::python::scope().attr("tenthDim") = blitz::tenthDim;
  boost::python::scope().attr("eleventhDim") = blitz::eleventhDim;

  //this maps the blitz ordering schemes
  bind_storages(1);
  bind_storages(2);
  bind_storages(3);
  bind_storages(4);
  bind_storages(5);
  bind_storages(6);
  bind_storages(7);
  bind_storages(8);
  bind_storages(9);
  bind_storages(10);
  bind_storages(11);

}
