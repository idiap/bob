/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Thu  2 Dec 07:41:43 2010 
 *
 * @brief Mappings between ndarray types and blitz::Array<> supported types.
 */

#ifndef TORCH_CORE_PYTHON_TYPEMAPPER_H 
#define TORCH_CORE_PYTHON_TYPEMAPPER_H

#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include <boost/format.hpp>
#include <string>
#include <map>

namespace Torch { namespace python {

  /**
   * This map allows conversion from the C type code (enum) into the python 
   * type code (single char) or type name (string) and vice-versa. We
   * instantiate a static variable of this type and use it throughout the code.
   */
  struct TypeMapper { 

    public:
      /**
       * Constructor
       */
      TypeMapper();

      /** 
       * Conversion from Numpy C enum to Numpy single character typecode
       */
      const std::string& enum_to_code(NPY_TYPES t) const;

      /**
       * Conversion from Numpy C enum to string description
       */
      const std::string& enum_to_name(NPY_TYPES t) const; 

      /**
       * Conversion from Numpy C enum to blitz::Array<T,N>::T typename
       */
      const std::string& enum_to_blitzT(NPY_TYPES t) const; 

      /**
       * Converts from Numpy C enum to the size of the scalar
       */
      size_t enum_to_scalar_size(NPY_TYPES t) const;

      /**
       * Converts from Numpy C enum to the base type of the scalar
       */
      char enum_to_scalar_base(NPY_TYPES t) const;

      /**
       * Conversion from C++ type to Numpy C enum
       */
      template <typename T> NPY_TYPES type_to_enum(void) const {
        boost::format s("Instantiation of function type_to_enum() for class '%s' is not yet implemented.", typeid(T).name());
        PyErr_SetString(PyExc_NotImplementedError, s.str().c_str());
        throw boost::python::error_already_set();
      }

      /**
       * Conversion from C++ type to Numpy typecode
       */
      template <typename T> const std::string& type_to_typecode(void) const 
      { return enum_to_code(type_to_enum<T>()); }

      /**
       * Conversion from C++ type to Numpy typename
       */
      template <typename T> const std::string& type_to_typename(void) const 
      { return enum_to_name(type_to_enum<T>()); }

    private:
      std::string bind(const char* base, int size) const;
      std::string bind_typename(const char* base, const char* type, int size) const;
      const std::string& get(const std::map<NPY_TYPES, std::string>& dict,
          NPY_TYPES t) const;

      template <typename T> const typename T::mapped_type& get_raise
        (const T& dict, NPY_TYPES t) const {
          typename T::const_iterator it = dict.find(t); 
          if (it == dict.end()) {
            boost::format f("value not listed in internal mapper (%s - %s)");
            f % this->get(this->m_c_to_typecode, t);
            f % this->get(this->m_c_to_typename, t);
            PyErr_SetString(PyExc_ValueError, f.str().c_str()); 
            throw boost::python::error_already_set(); 
          }
          return it->second;
        }

    private:
      std::map<NPY_TYPES, std::string> m_c_to_typecode;
      std::map<NPY_TYPES, std::string> m_c_to_typename;
      std::map<NPY_TYPES, std::string> m_c_to_blitz;
      std::map<NPY_TYPES, size_t> m_scalar_size;
      std::map<NPY_TYPES, char> m_scalar_base;
  }; 

  template <> NPY_TYPES TypeMapper::type_to_enum<bool>(void) const; 
  template <> NPY_TYPES TypeMapper::type_to_enum<signed char>(void) const;
  template <> NPY_TYPES TypeMapper::type_to_enum<unsigned char>(void) const; 
  template <> NPY_TYPES TypeMapper::type_to_enum<short>(void) const;
  template <> NPY_TYPES TypeMapper::type_to_enum<unsigned short>(void) const; 
  template <> NPY_TYPES TypeMapper::type_to_enum<int>(void) const; 
  template <> NPY_TYPES TypeMapper::type_to_enum<unsigned int>(void) const; 
  template <> NPY_TYPES TypeMapper::type_to_enum<long>(void) const;
  template <> NPY_TYPES TypeMapper::type_to_enum<unsigned long>(void) const;
  template <> NPY_TYPES TypeMapper::type_to_enum<long long>(void) const;
  template <> NPY_TYPES TypeMapper::type_to_enum<unsigned long long>(void) const;
  template <> NPY_TYPES TypeMapper::type_to_enum<float>(void) const;
  template <> NPY_TYPES TypeMapper::type_to_enum<double>(void) const;
  template <> NPY_TYPES TypeMapper::type_to_enum<long double>(void) const; 
  template <> NPY_TYPES TypeMapper::type_to_enum<std::complex<float> >(void) const;
  template <> NPY_TYPES TypeMapper::type_to_enum<std::complex<double> >(void) const;
  template <> NPY_TYPES TypeMapper::type_to_enum<std::complex<long double> >(void) const; 

  extern struct TypeMapper TYPEMAP;

}} //ends namespace Torch::python

#endif /* TORCH_CORE_PYTHON_TYPEMAPPER_H */
