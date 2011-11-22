/**
 * @file python/core/core/python/mapstring.h
 * @date Fri Jul 22 20:13:49 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Custom to/from C++ conversion for std::map with std::string keys
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

#ifndef TORCH_CORE_PYTHON_MAPSTRING_H 
#define TORCH_CORE_PYTHON_MAPSTRING_H

#include <boost/python.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include "core/python/exception.h"
#include "core/python/map_container.h"
#include <string>

namespace Torch { namespace python {

  /**
   * Main method to defined bindings for std::map<std::string,T>. Please note that the
   * type T should have a defined operator==() in explicit or implicit way.
   */
  template <typename T> void mapstring(const char* basename) {

    //defines basic bindings for methods manipulating std::map<std::string, T>.
    boost::python::class_<std::map<std::string, T> >(basename)
      .def(boost::python::map_indexing_suite<std::map<std::string, T> >());

    //register the python::dict to C++ std::map<std::string, T> auto conversion.
    from_python_dict<std::map<std::string, T> >();

  }

  /**
   * This trick allows for maps in which the contained element does not
   * provide a operator==().
   */
  template <class T> class no_compare_indexing_suite : public boost::python::map_indexing_suite<T, false, no_compare_indexing_suite<T> > {
    public:
      static bool contains(T &container, typename T::key_type const &key) { 
        PYTHON_ERROR(NotImplementedError, "containment checking not supported on this container"); 
      }
  };

  template <typename T> void mapstring_no_compare(const char* basename) {

    //defines basic bindings for methods manipulating std::maps.
    boost::python::class_<std::map<std::string, T> >(basename)
      .def(no_compare_indexing_suite<std::map<std::string, T> >());

    //register the python::dict to C++ std::map<std::string,T> auto conversion.
    from_python_dict<std::map<std::string,T> >();

  }

}}

#endif /* TORCH_CORE_PYTHON_MAP_H */

