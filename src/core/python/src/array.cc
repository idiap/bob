/**
 * @file python/src/array.cc
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 *
 * @brief blitz::Array<> to and from python converters for arrays
 */

#include "core/python/array.h"

Torch::python::TypeMapper Torch::python::TYPEMAP;

Torch::python::TypeMapper::TypeMapper() {
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

template <> NPY_TYPES Torch::python::TypeMapper::type_to_enum<bool>(void) const 
{ return NPY_BOOL; }
template <> NPY_TYPES Torch::python::TypeMapper::type_to_enum<signed char>(void) const 
{ return NPY_BYTE; }
template <> NPY_TYPES Torch::python::TypeMapper::type_to_enum<unsigned char>(void) const 
{ return NPY_UBYTE; }
template <> NPY_TYPES Torch::python::TypeMapper::type_to_enum<short>(void) const 
{ return NPY_SHORT; }
template <> NPY_TYPES Torch::python::TypeMapper::type_to_enum<unsigned short>(void) const 
{ return NPY_USHORT; }
template <> NPY_TYPES Torch::python::TypeMapper::type_to_enum<int>(void) const 
{ return NPY_INT; }
template <> NPY_TYPES Torch::python::TypeMapper::type_to_enum<unsigned int>(void) const 
{ return NPY_UINT; }
template <> NPY_TYPES Torch::python::TypeMapper::type_to_enum<long>(void) const
{ return NPY_LONG; }
template <> NPY_TYPES Torch::python::TypeMapper::type_to_enum<unsigned long>(void) const
{ return NPY_ULONG; }
template <> NPY_TYPES Torch::python::TypeMapper::type_to_enum<long long>(void) const
{ return NPY_LONGLONG; }
template <> NPY_TYPES Torch::python::TypeMapper::type_to_enum<unsigned long long>(void) const
{ return NPY_ULONGLONG; }
template <> NPY_TYPES Torch::python::TypeMapper::type_to_enum<float>(void) const
{ return NPY_FLOAT; }
template <> NPY_TYPES Torch::python::TypeMapper::type_to_enum<double>(void) const 
{ return NPY_DOUBLE; }
template <> NPY_TYPES Torch::python::TypeMapper::type_to_enum<long double>(void) const 
{ return NPY_LONGDOUBLE; }
template <> NPY_TYPES Torch::python::TypeMapper::type_to_enum<std::complex<float> >(void) const
{ return NPY_CFLOAT; }
template <> NPY_TYPES Torch::python::TypeMapper::type_to_enum<std::complex<double> >(void) const 
{ return NPY_CDOUBLE; }
template <> NPY_TYPES Torch::python::TypeMapper::type_to_enum<std::complex<long double> >(void) const 
{ return NPY_CLONGDOUBLE; }

const std::string& Torch::python::TypeMapper::enum_to_code(NPY_TYPES t) const 
{ return get(this->m_c_to_typecode, t); }

const std::string& Torch::python::TypeMapper::enum_to_name(NPY_TYPES t) const 
{ return get(this->m_c_to_typename, t); }

const std::string& Torch::python::TypeMapper::enum_to_blitzT(NPY_TYPES t) const 
{ return get_raise(this->m_c_to_blitz, t); }

size_t Torch::python::TypeMapper::enum_to_scalar_size(NPY_TYPES t) const
{ return get_raise(this->m_scalar_size, t); }

char Torch::python::TypeMapper::enum_to_scalar_base(NPY_TYPES t) const
{ return get_raise(this->m_scalar_base, t); }

bool Torch::python::TypeMapper::are_equivalent(NPY_TYPES i1, NPY_TYPES i2) const {
  return (i1 == i2) || 
    ((enum_to_scalar_base(i1) == enum_to_scalar_base(i2)) && 
     (enum_to_scalar_size(i1) == enum_to_scalar_size(i2)));
}

std::string Torch::python::TypeMapper::bind(const char* base, int size) const {
  boost::format f("%s%d");
  f % base % (8*size);
  return f.str();
}

std::string Torch::python::TypeMapper::bind_typename
(const char* base, const char* type, int size) const {
  boost::format f("%s (%s%d)");
  f % base % type % (8*size);
  return f.str();
}

const std::string& Torch::python::TypeMapper::get
(const std::map<NPY_TYPES, std::string>& dict, NPY_TYPES t) const {
  static const std::string EMPTY("not mapped");
  std::map<NPY_TYPES, std::string>::const_iterator it = dict.find(t); 
  if (it == dict.end()) return EMPTY;
  return it->second;
}

NPY_TYPES Torch::python::type(boost::python::numeric::array a) {
  return NPY_TYPES(PyArray_TYPE(a.ptr())); 
}

std::string Torch::python::equivalent_scalar(boost::python::numeric::array a) {
  boost::format f("%s_%d");
  f % TYPEMAP.enum_to_blitzT(Torch::python::type(a)) % PyArray_NDIM(a.ptr());
  return f.str();
}

size_t Torch::python::rank(boost::python::numeric::array a) {
  return PyArray_NDIM(a.ptr());
}

void Torch::python::check_rank(boost::python::numeric::array a, size_t expected_rank) {
  if (rank(a) != expected_rank) {
    boost::format err("expected array with rank %d, got instead %d");
    err % expected_rank % rank(a);
    PyErr_SetString(PyExc_RuntimeError, err.str().c_str());
    boost::python::throw_error_already_set();
  }
}

void Torch::python::check_is_array(boost::python::object o) {
  if(!PyArray_Check(o.ptr())){
    PyErr_SetString(PyExc_ValueError, "expected ndarray");
    boost::python::throw_error_already_set();
  }
}

boost::python::numeric::array Torch::python::astype(boost::python::numeric::array a, const::std::string& t) {
  return (boost::python::numeric::array) a.astype(t);
}

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

PyObject* Torch::python::create_numpy_array(int N, npy_intp* dimensions,
    NPY_TYPES tp) {
  return PyArray_SimpleNew(N, dimensions, tp);
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

#define bind_storages(N) bind_c_storage<N>(); bind_fortran_storage<N>();

void bind_core_array() {
  /**
   * Resets the module and type used by boost::python to numpy
   */
  boost::python::numeric::array::set_module_and_type("numpy", "ndarray");

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

  boost::python::def("equivalent_scalar", &Torch::python::equivalent_scalar, (boost::python::arg("array")), "Returns the python-binding blitz::Array equivalent name for the given numpy array");
 
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
