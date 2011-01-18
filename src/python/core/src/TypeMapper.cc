/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Thu  2 Dec 07:56:22 2010 
 *
 * @brief Defines the Torch::python::TypeMapper for mapping between ndarray
 * element types and torch supported blitz::Array<> ones. 
 */

#include "core/python/TypeMapper.h"

namespace bp = boost::python;
namespace tp = Torch::python;

tp::TypeMapper tp::TYPEMAP;

tp::TypeMapper::TypeMapper() {
  m_c_to_typecode[NPY_BOOL] = "?";
  m_c_to_typecode[NPY_BYTE] = "b";
  m_c_to_typecode[NPY_UBYTE] = "B";
  m_c_to_typecode[NPY_SHORT] = "h";
  m_c_to_typecode[NPY_USHORT] = "H";
  m_c_to_typecode[NPY_INT] = "i";
  m_c_to_typecode[NPY_UINT] = "I";
  m_c_to_typecode[NPY_LONG] = "l";
  m_c_to_typecode[NPY_ULONG] = "L";
  m_c_to_typecode[NPY_LONGLONG] = "q";
  m_c_to_typecode[NPY_ULONGLONG] = "Q";
  m_c_to_typecode[NPY_FLOAT] = "f";
  m_c_to_typecode[NPY_DOUBLE] = "d";
  m_c_to_typecode[NPY_LONGDOUBLE] = "g";
  m_c_to_typecode[NPY_CFLOAT] = "F"; 
  m_c_to_typecode[NPY_CDOUBLE] = "D";
  m_c_to_typecode[NPY_CLONGDOUBLE] = "G";
  m_c_to_typecode[NPY_OBJECT] = "O";
  m_c_to_typecode[NPY_STRING] = "S";
  m_c_to_typecode[NPY_UNICODE] = "U";
  m_c_to_typecode[NPY_VOID] = "V";
  m_c_to_typecode[NPY_NTYPES] = "NPY_NTYPES"; ///< no conversion found
  m_c_to_typecode[NPY_NOTYPE] = "NPY_NOTYPE"; ///< no conversion found
  m_c_to_typecode[NPY_CHAR] = "S1";
  m_c_to_typecode[NPY_USERDEF] = "NPY_USERDEF"; ///< no conversion found

  m_c_to_typename[NPY_BOOL] = "bool";
  m_c_to_typename[NPY_BYTE] = bind_typename("signed char", "int", sizeof(char));
  m_c_to_typename[NPY_UBYTE] = bind_typename("unsigned char", "uint", sizeof(char));
  m_c_to_typename[NPY_SHORT] = bind_typename("short", "int", sizeof(short));
  m_c_to_typename[NPY_USHORT] = bind_typename("unsigned short", "uint", sizeof(unsigned short));
  m_c_to_typename[NPY_INT] = bind_typename("integer", "int", sizeof(int));
  m_c_to_typename[NPY_UINT] = bind_typename("unsigned integer", "uint", sizeof(unsigned int));
  m_c_to_typename[NPY_LONG] = bind_typename("long integer", "int", sizeof(long));
  m_c_to_typename[NPY_ULONG] = bind_typename("unsigned long integer", "uint", sizeof(unsigned long));
  m_c_to_typename[NPY_LONGLONG] = bind_typename("long long integer", "int", sizeof(long long));
  m_c_to_typename[NPY_ULONGLONG] = bind_typename("unsigned long long integer", "uint", sizeof(unsigned long long));
  m_c_to_typename[NPY_FLOAT] = bind_typename("single precision", "float", sizeof(float));
  m_c_to_typename[NPY_DOUBLE] = bind_typename("double precision", "float", sizeof(double));
  m_c_to_typename[NPY_LONGDOUBLE] = bind_typename("long precision", "float", sizeof(long double));
  m_c_to_typename[NPY_CFLOAT] = bind_typename("complex single precision", "complex", sizeof(std::complex<float>));
  m_c_to_typename[NPY_CDOUBLE] = bind_typename("complex double precision", "complex", sizeof(std::complex<double>));
  m_c_to_typename[NPY_CLONGDOUBLE] = bind_typename("complex long double precision", "complex", sizeof(std::complex<long double>));
  m_c_to_typename[NPY_OBJECT] = "object";
  m_c_to_typename[NPY_STRING] = "string";
  m_c_to_typename[NPY_UNICODE] = "unicode";
  m_c_to_typename[NPY_VOID] = "void";
  m_c_to_typename[NPY_NTYPES] = "enum types";
  m_c_to_typename[NPY_NOTYPE] = "no type"; 
  m_c_to_typename[NPY_CHAR] = "character";
  m_c_to_typename[NPY_USERDEF] = "user defined";

  m_c_to_blitz[NPY_BOOL] = "bool";
  m_c_to_blitz[NPY_BYTE] = bind("int", sizeof(char));
  m_c_to_blitz[NPY_UBYTE] = bind("uint", sizeof(unsigned char));
  m_c_to_blitz[NPY_SHORT] = bind("int", sizeof(short));
  m_c_to_blitz[NPY_USHORT] = bind("uint", sizeof(unsigned short));
  m_c_to_blitz[NPY_INT] = bind("int", sizeof(int));
  m_c_to_blitz[NPY_UINT] = bind("uint", sizeof(unsigned int));
  m_c_to_blitz[NPY_LONG] = bind("int", sizeof(long));
  m_c_to_blitz[NPY_ULONG] = bind("uint", sizeof(unsigned long));
  m_c_to_blitz[NPY_LONGLONG] = bind("int", sizeof(long long));
  m_c_to_blitz[NPY_ULONGLONG] = bind("uint", sizeof(unsigned long long));
  m_c_to_blitz[NPY_FLOAT] = bind("float", sizeof(float));
  m_c_to_blitz[NPY_DOUBLE] = bind("float", sizeof(double));
  m_c_to_blitz[NPY_LONGDOUBLE] = bind("float", sizeof(long double));
  m_c_to_blitz[NPY_CFLOAT] = bind("complex", sizeof(std::complex<float>));
  m_c_to_blitz[NPY_CDOUBLE] = bind("complex", sizeof(std::complex<double>));
  m_c_to_blitz[NPY_CLONGDOUBLE] = bind("complex", sizeof(std::complex<long double>));

  m_scalar_size[NPY_BOOL] = sizeof(bool);
  m_scalar_size[NPY_BYTE] = sizeof(char);
  m_scalar_size[NPY_UBYTE] = sizeof(unsigned char);
  m_scalar_size[NPY_SHORT] = sizeof(short);
  m_scalar_size[NPY_USHORT] = sizeof(unsigned short);
  m_scalar_size[NPY_INT] = sizeof(int);
  m_scalar_size[NPY_UINT] = sizeof(unsigned int);
  m_scalar_size[NPY_LONG] = sizeof(long);
  m_scalar_size[NPY_ULONG] = sizeof(unsigned long);
  m_scalar_size[NPY_LONGLONG] = sizeof(long long);
  m_scalar_size[NPY_ULONGLONG] = sizeof(unsigned long long);
  m_scalar_size[NPY_FLOAT] = sizeof(float);
  m_scalar_size[NPY_DOUBLE] = sizeof(double);
  m_scalar_size[NPY_LONGDOUBLE] = sizeof(long double);
  m_scalar_size[NPY_CFLOAT] = sizeof(std::complex<float>);
  m_scalar_size[NPY_CDOUBLE] = sizeof(std::complex<double>);
  m_scalar_size[NPY_CLONGDOUBLE] = sizeof(std::complex<long double>);

  m_scalar_base[NPY_BOOL] = 'b'; 
  m_scalar_base[NPY_BYTE] = 'i'; 
  m_scalar_base[NPY_UBYTE] = 'i'; 
  m_scalar_base[NPY_SHORT] = 'i'; 
  m_scalar_base[NPY_USHORT] = 'i';
  m_scalar_base[NPY_INT] = 'i';
  m_scalar_base[NPY_UINT] = 'i';
  m_scalar_base[NPY_LONG] = 'i';
  m_scalar_base[NPY_ULONG] = 'i';
  m_scalar_base[NPY_LONGLONG] = 'i';
  m_scalar_base[NPY_ULONGLONG] = 'i';
  m_scalar_base[NPY_FLOAT] = 'f';
  m_scalar_base[NPY_DOUBLE] = 'f';
  m_scalar_base[NPY_LONGDOUBLE] = 'f';
  m_scalar_base[NPY_CFLOAT] = 'c';
  m_scalar_base[NPY_CDOUBLE] = 'c';
  m_scalar_base[NPY_CLONGDOUBLE] = 'c';
}

template <> NPY_TYPES tp::TypeMapper::type_to_enum<bool>(void) const 
{ return NPY_BOOL; }
template <> NPY_TYPES tp::TypeMapper::type_to_enum<signed char>(void) const 
{ return NPY_BYTE; }
template <> NPY_TYPES tp::TypeMapper::type_to_enum<unsigned char>(void) const 
{ return NPY_UBYTE; }
template <> NPY_TYPES tp::TypeMapper::type_to_enum<short>(void) const 
{ return NPY_SHORT; }
template <> NPY_TYPES tp::TypeMapper::type_to_enum<unsigned short>(void) const 
{ return NPY_USHORT; }
template <> NPY_TYPES tp::TypeMapper::type_to_enum<int>(void) const 
{ return NPY_INT; }
template <> NPY_TYPES tp::TypeMapper::type_to_enum<unsigned int>(void) const 
{ return NPY_UINT; }
template <> NPY_TYPES tp::TypeMapper::type_to_enum<long>(void) const
{ return NPY_LONG; }
template <> NPY_TYPES tp::TypeMapper::type_to_enum<unsigned long>(void) const
{ return NPY_ULONG; }
template <> NPY_TYPES tp::TypeMapper::type_to_enum<long long>(void) const
{ return NPY_LONGLONG; }
template <> NPY_TYPES tp::TypeMapper::type_to_enum<unsigned long long>(void) const
{ return NPY_ULONGLONG; }
template <> NPY_TYPES tp::TypeMapper::type_to_enum<float>(void) const
{ return NPY_FLOAT; }
template <> NPY_TYPES tp::TypeMapper::type_to_enum<double>(void) const 
{ return NPY_DOUBLE; }
template <> NPY_TYPES tp::TypeMapper::type_to_enum<long double>(void) const 
{ return NPY_LONGDOUBLE; }
template <> NPY_TYPES tp::TypeMapper::type_to_enum<std::complex<float> >(void) const
{ return NPY_CFLOAT; }
template <> NPY_TYPES tp::TypeMapper::type_to_enum<std::complex<double> >(void) const 
{ return NPY_CDOUBLE; }
template <> NPY_TYPES tp::TypeMapper::type_to_enum<std::complex<long double> >(void) const 
{ return NPY_CLONGDOUBLE; }

const std::string& tp::TypeMapper::enum_to_code(NPY_TYPES t) const 
{ return get(this->m_c_to_typecode, t); }

const std::string& tp::TypeMapper::enum_to_name(NPY_TYPES t) const 
{ return get(this->m_c_to_typename, t); }

const std::string& tp::TypeMapper::enum_to_blitzT(NPY_TYPES t) const 
{ return get_raise(this->m_c_to_blitz, t); }

size_t tp::TypeMapper::enum_to_scalar_size(NPY_TYPES t) const
{ return get_raise(this->m_scalar_size, t); }

char tp::TypeMapper::enum_to_scalar_base(NPY_TYPES t) const
{ return get_raise(this->m_scalar_base, t); }

std::string tp::TypeMapper::bind(const char* base, int size) const {
  boost::format f("%s%d");
  f % base % (8*size);
  return f.str();
}

std::string tp::TypeMapper::bind_typename
(const char* base, const char* type, int size) const {
  boost::format f("%s (%s%d)");
  f % base % type % (8*size);
  return f.str();
}

const std::string& tp::TypeMapper::get
(const std::map<NPY_TYPES, std::string>& dict, NPY_TYPES t) const {
  static const std::string EMPTY("not mapped");
  std::map<NPY_TYPES, std::string>::const_iterator it = dict.find(t); 
  if (it == dict.end()) return EMPTY;
  return it->second;
}
