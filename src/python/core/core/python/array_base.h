/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Wed  9 Mar 17:02:34 2011 
 *
 * @brief Instantiates the base python types
 */

#ifndef TORCH_CORE_PYTHON_ARRAY_BASE_H
#define TORCH_CORE_PYTHON_ARRAY_BASE_H

#include <boost/make_shared.hpp>
#include <boost/python.hpp>
#include <boost/format.hpp>
#include <blitz/array.h>
#include <stdint.h>

#include "core/array_type.h"
#include "core/blitz_compat.h"

namespace Torch { namespace python {

  /**
   * This struct will declare the several bits that are part of the Torch
   * blitz::Array<T,N> python bindings.
   */
  template<typename T, int N> struct array {

    public:
      //declares a few typedefs to make it easier to write code
      typedef typename blitz::Array<T,N> array_type;
      typedef typename blitz::TinyVector<int,N> shape_type;
      typedef typename blitz::TinyVector<blitz::diffType,N> stride_type;
      typedef typename blitz::GeneralArrayStorage<N> storage_type;
      typedef array_type& (array_type::*inplace_const_op)(const T&);
      typedef array_type& (array_type::*inplace_array_op)(const array_type&);
      typedef boost::python::class_<array_type, boost::shared_ptr<array_type> > py_type;
      typedef boost::shared_ptr<py_type> spy_type;

    private: //our representation
      std::string m_element_type_str;
      std::string m_blitz_type_str;
      spy_type m_class;

    public:

      /**
       * Returns a pointer to my internal boost::python class bindings
       */
      spy_type object () { return m_class; }

    public:
      /**
       * The constructor does the basics for the initialization of objects of
       * this class. It will NOT call all boost methods to build the bindings.
       */
      array(const char* tname) {

        m_element_type_str = tname;
        boost::format blitz_name("blitz::Array<%s,%d>");
        blitz_name % tname % N;
        m_blitz_type_str = blitz_name.str();

      }

      /**
       * This method effectively creates the bindings and has to be called for
       * all supported types inside the module where the class names should be
       * attached to.
       */
      void bind() {
        boost::format class_name("%s_%d");
        class_name % m_element_type_str % N;
        boost::format class_doc("Objects of this class are a pythonic representation of %s. Please refer to the blitz::Array manual for more details on the array class and its usage. The help messages attached to each member function of this binding are just for quick-referencing. (N.B. Dimensions in C-arrays are zero-indexed. The first dimension is 0, the second is 1, etc. Use the helpers 'firstDim', 'secondDim', etc to help you keeping your code clear.)");
        class_doc % m_blitz_type_str;

        //base class creation
        m_class = boost::make_shared<py_type>(class_name.str().c_str(), class_doc.str().c_str(), boost::python::init<>("Initializes an empty array"));

        //bind some reference variables
        m_class->def_readonly("cxx_element_typename", &m_element_type_str); 
        m_class->def_readonly("cxx_blitz_typename", &m_blitz_type_str); 
      }

  };

  /**
   * Given the element type and the number of dimensions, return the correct
   * bound boost python class_ object.
   */
  boost::python::object array_class(Torch::core::array::ElementType eltype,
      int rank);

  /**
   * Gets the class using a proper typename
   */
  template <typename T, int N> inline
    boost::python::object array_class() {
      return array_class(Torch::core::array::getElementType<T>(), N);
    }

  /**
   * The following variables are globals that contain our bindings
   */
  extern array<bool, 1> bool_1;
  extern array<bool, 2> bool_2;
  extern array<bool, 3> bool_3;
  extern array<bool, 4> bool_4;
  
  extern array<int8_t, 1> int8_1;
  extern array<int8_t, 2> int8_2;
  extern array<int8_t, 3> int8_3;
  extern array<int8_t, 4> int8_4;
  
  extern array<int16_t, 1> int16_1;
  extern array<int16_t, 2> int16_2;
  extern array<int16_t, 3> int16_3;
  extern array<int16_t, 4> int16_4;
  
  extern array<int32_t, 1> int32_1;
  extern array<int32_t, 2> int32_2;
  extern array<int32_t, 3> int32_3;
  extern array<int32_t, 4> int32_4;
  
  extern array<int64_t, 1> int64_1;
  extern array<int64_t, 2> int64_2;
  extern array<int64_t, 3> int64_3;
  extern array<int64_t, 4> int64_4;
  
  extern array<uint8_t, 1> uint8_1;
  extern array<uint8_t, 2> uint8_2;
  extern array<uint8_t, 3> uint8_3;
  extern array<uint8_t, 4> uint8_4;
  
  extern array<uint16_t, 1> uint16_1;
  extern array<uint16_t, 2> uint16_2;
  extern array<uint16_t, 3> uint16_3;
  extern array<uint16_t, 4> uint16_4;
  
  extern array<uint32_t, 1> uint32_1;
  extern array<uint32_t, 2> uint32_2;
  extern array<uint32_t, 3> uint32_3;
  extern array<uint32_t, 4> uint32_4;
  
  extern array<uint64_t, 1> uint64_1;
  extern array<uint64_t, 2> uint64_2;
  extern array<uint64_t, 3> uint64_3;
  extern array<uint64_t, 4> uint64_4;
  
  extern array<float, 1> float32_1;
  extern array<float, 2> float32_2;
  extern array<float, 3> float32_3;
  extern array<float, 4> float32_4;
  
  extern array<double, 1> float64_1;
  extern array<double, 2> float64_2;
  extern array<double, 3> float64_3;
  extern array<double, 4> float64_4;
  
  extern array<long double, 1> float128_1;
  extern array<long double, 2> float128_2;
  extern array<long double, 3> float128_3;
  extern array<long double, 4> float128_4;
  
  extern array<std::complex<float>, 1> complex64_1;
  extern array<std::complex<float>, 2> complex64_2;
  extern array<std::complex<float>, 3> complex64_3;
  extern array<std::complex<float>, 4> complex64_4;
  
  extern array<std::complex<double>, 1> complex128_1;
  extern array<std::complex<double>, 2> complex128_2;
  extern array<std::complex<double>, 3> complex128_3;
  extern array<std::complex<double>, 4> complex128_4;
  
  extern array<std::complex<long double>, 1> complex256_1;
  extern array<std::complex<long double>, 2> complex256_2;
  extern array<std::complex<long double>, 3> complex256_3;
  extern array<std::complex<long double>, 4> complex256_4;

}}

#endif /* TORCH_CORE_PYTHON_ARRAY_BASE_H */
