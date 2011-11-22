/**
 * @file python/core/core/python/vector.h
 * @date Tue Jun 21 17:28:25 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * Custom to/from C++ conversion for std::vector and std::list
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

#ifndef TORCH_CORE_PYTHON_VECTOR_H 
#define TORCH_CORE_PYTHON_VECTOR_H

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include "core/python/exception.h"
#include "core/python/container.h"

namespace Torch { namespace python {

  /**
   * Main method to defined bindings for std::vector<T>. Please note that the
   * type T should have a defined operator==() in explicit or implicit way.
   */
  template <typename T> void vector(const char* basename) {

    //defines basic bindings for methods manipulating std::vectors.
    boost::python::class_<std::vector<T> >(basename)
      .def(boost::python::vector_indexing_suite<std::vector<T> >());

    //register the python::sequence to C++ std::vector<T> auto conversion.
    from_python_sequence<std::vector<T>, variable_capacity_policy>();

  }

  /**
   * This trick allows for vectors in which the contained element does not
   * provide a operator==().
   */
  template <class T> class no_compare_indexing_suite : public boost::python::vector_indexing_suite<T, false, no_compare_indexing_suite<T> > {
    public:
      static bool contains(T &container, typename T::value_type const &key) { 
        PYTHON_ERROR(NotImplementedError, "containment checking not supported on this container"); 
      }
  };

  template <typename T> void vector_no_compare(const char* basename) {

    //defines basic bindings for methods manipulating std::vectors.
    boost::python::class_<std::vector<T> >(basename)
      .def(no_compare_indexing_suite<std::vector<T> >());

    //register the python::sequence to C++ std::vector<T> auto conversion.
    from_python_sequence<std::vector<T>, variable_capacity_policy>();

  }

}}

#endif /* TORCH_CORE_PYTHON_VECTOR_H */

