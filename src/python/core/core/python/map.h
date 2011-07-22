/**
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Fri 22 Jul 2011 15:05:48 CEST
 *
 * Custom to/from C++ conversion for std::map
 */

#ifndef TORCH_CORE_PYTHON_MAP_H 
#define TORCH_CORE_PYTHON_MAP_H

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
  template <typename T> void map(const char* basename) {

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
  // TODO
  template <class T> class no_compare_indexing_suite : public boost::python::map_indexing_suite<T, false, no_compare_indexing_suite<T> > {
    public:
      static bool contains(T &container, typename T::key_type const &key) { 
        PYTHON_ERROR(NotImplementedError, "containment checking not supported on this container"); 
      }
  };

  template <typename T> void map_no_compare(const char* basename) {

    //defines basic bindings for methods manipulating std::maps.
    boost::python::class_<std::map<std::string, T> >(basename)
      .def(no_compare_indexing_suite<std::map<std::string, T> >());

    //register the TODO: python::sequence to C++ std::map<std::string,T> auto conversion.
    // TODO: from_python_dict ?
    from_python_dict<std::map<std::string,T> >();

  }

}}

#endif /* TORCH_CORE_PYTHON_MAP_H */

