/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Tue 27 Sep 12:36:44 2011
 *
 * @brief Implementation of a few blitz::Array<> => numpy.ndarray helpers.
 */

#include "core/python/bzhelper.h"

namespace bp = boost::python;
namespace tp = Torch::python;

template <> int tp::type_to_num<bool>(void) 
{ return NPY_BOOL; }
template <> int tp::type_to_num<signed char>(void) 
{ return NPY_BYTE; }
template <> int tp::type_to_num<unsigned char>(void) 
{ return NPY_UBYTE; }
template <> int tp::type_to_num<short>(void) 
{ return NPY_SHORT; }
template <> int tp::type_to_num<unsigned short>(void) 
{ return NPY_USHORT; }
template <> int tp::type_to_num<int>(void) 
{ return NPY_INT; }
template <> int tp::type_to_num<unsigned int>(void) 
{ return NPY_UINT; }
template <> int tp::type_to_num<long>(void)
{ return NPY_LONG; }
template <> int tp::type_to_num<unsigned long>(void)
{ return NPY_ULONG; }
template <> int tp::type_to_num<long long>(void)
{ return NPY_LONGLONG; }
template <> int tp::type_to_num<unsigned long long>(void)
{ return NPY_ULONGLONG; }
template <> int tp::type_to_num<float>(void)
{ return NPY_FLOAT; }
template <> int tp::type_to_num<double>(void) 
{ return NPY_DOUBLE; }
template <> int tp::type_to_num<long double>(void) 
{ return NPY_LONGDOUBLE; }
template <> int tp::type_to_num<std::complex<float> >(void)
{ return NPY_CFLOAT; }
template <> int tp::type_to_num<std::complex<double> >(void) 
{ return NPY_CDOUBLE; }
template <> int tp::type_to_num<std::complex<long double> >(void) 
{ return NPY_CLONGDOUBLE; }

tp::dtype::dtype(const bp::object& name): _m(0) {
  PyArray_DescrConverter(name.ptr(), &_m);
}

tp::dtype::dtype(const tp::dtype& other): _m(other._m) {
}

tp::dtype::~dtype() {
}
