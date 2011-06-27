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
#include <boost/preprocessor.hpp>
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
      array() {

        m_element_type_str = Torch::core::array::stringize<T>();
        boost::format blitz_name("blitz::Array<%s,%d>");
        blitz_name % m_element_type_str % N;
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
# define BOOST_PP_LOCAL_LIMITS (1, TORCH_MAX_DIM)
# define BOOST_PP_LOCAL_MACRO(D) \
  extern array<bool, D> BOOST_PP_CAT(bool_,D);\
  extern array<int8_t, D> BOOST_PP_CAT(int8_,D);\
  extern array<int16_t, D> BOOST_PP_CAT(int16_,D);\
  extern array<int32_t, D> BOOST_PP_CAT(int32_,D);\
  extern array<int64_t, D> BOOST_PP_CAT(int64_,D);\
  extern array<uint8_t, D> BOOST_PP_CAT(uint8_,D);\
  extern array<uint16_t, D> BOOST_PP_CAT(uint16_,D);\
  extern array<uint32_t, D> BOOST_PP_CAT(uint32_,D);\
  extern array<uint64_t, D> BOOST_PP_CAT(uint64_,D);\
  extern array<float, D> BOOST_PP_CAT(float32_,D);\
  extern array<double, D> BOOST_PP_CAT(float64_,D);\
  extern array<long double, D> BOOST_PP_CAT(float128_,D);\
  extern array<std::complex<float>, D> BOOST_PP_CAT(complex64_,D);\
  extern array<std::complex<double>, D> BOOST_PP_CAT(complex128_,D);\
  extern array<std::complex<long double>, D> BOOST_PP_CAT(complex256_,D);
#include BOOST_PP_LOCAL_ITERATE()

}}

#endif /* TORCH_CORE_PYTHON_ARRAY_BASE_H */
