/**
 * @file src/python/core/src/array.cc
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 *
 * @brief blitz::Array<> to and from python converters for arrays
 */

#include <boost/shared_ptr.hpp>
#include <boost/python.hpp>
#include <blitz/array.h>

#include "core/python/ndarray.h"

namespace bp = boost::python;

template <int N> 
bp::class_<blitz::GeneralArrayStorage<N>,
  boost::shared_ptr<blitz::GeneralArrayStorage<N> > > bind_c_storage() {

  typedef typename blitz::GeneralArrayStorage<N> storage_type;
  boost::format class_name("c_storage_%d");
  class_name % N;
  boost::format class_doc("Storages of this type can be used to force a certain storage style in an array. Use objects of this type to force a C-storage type for a %d-D array.");
  class_doc % N;
  bp::class_<storage_type, boost::shared_ptr<storage_type> > 
    retval(class_name.str().c_str(), class_doc.str().c_str(), bp::init<>());
  return retval;
}

template <int N>
bp::class_<blitz::FortranArray<N>,
  boost::shared_ptr<blitz::FortranArray<N> > > bind_fortran_storage() {
  typedef typename blitz::FortranArray<N> storage_type;
  boost::format class_name("fortran_storage_%d");
  class_name % N;
  boost::format class_doc("Storages of this type can be used to force a certain storage style in an array. Use objects of this type to force a Fortran-storage type for a %d-D array.");
  class_doc % N;
  bp::class_<storage_type, boost::shared_ptr<storage_type> > 
    retval(class_name.str().c_str(), class_doc.str().c_str(), bp::init<>());
  return retval;
}

#define bind_storages(N) bind_c_storage<N>(); bind_fortran_storage<N>();

void bind_array_storage() {
  bp::enum_<NPY_TYPES>("NPY_TYPES")
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
  bp::scope().attr("firstDim") = blitz::firstDim;
  bp::scope().attr("secondDim") = blitz::secondDim;
  bp::scope().attr("thirdDim") = blitz::thirdDim;
  bp::scope().attr("fourthDim") = blitz::fourthDim;
  bp::scope().attr("fifthDim") = blitz::fifthDim;
  bp::scope().attr("sixthDim") = blitz::sixthDim;
  bp::scope().attr("seventhDim") = blitz::seventhDim;
  bp::scope().attr("eighthDim") = blitz::eighthDim;
  bp::scope().attr("ninthDim") = blitz::ninthDim;
  bp::scope().attr("tenthDim") = blitz::tenthDim;
  bp::scope().attr("eleventhDim") = blitz::eleventhDim;

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
