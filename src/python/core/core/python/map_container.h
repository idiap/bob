/**
 * @file python/core/core/python/map_container.h
 * @date Fri Jul 22 16:13:23 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Generic container conversions for boost::python, extracted from: http://cctbx.cvs.sourceforge.net/viewvc/cctbx/scitbx/include/scitbx/stl/map_wrapper.h
 * version 1.16 (not touched for at least 5 years).
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

#ifndef TORCH_CORE_PYTHON_MAP_CONTAINER_H 
#define TORCH_CORE_PYTHON_MAP_CONTAINER_H

#include <boost/python.hpp>
#include <blitz/array.h>
#include "core/python/exception.h"
#include "core/array_copy.h"

namespace Torch { namespace python {

  //helps assignment with specific types
  template <typename T>
    struct assign {
      void operator() (std::map<std::string, T>& self, const std::string& key, const T& value) {
        self[key] = value;
      }
    };

  //assignment specialization for blitz::Array<V,N>
  template <>
    template <typename V, int N>
    struct assign<blitz::Array<V,N> > {
      void operator() (std::map<std::string, blitz::Array<V,N> >& self, const std::string& key, const blitz::Array<V,N>& value) {
        self[key].reference(Torch::core::array::ccopy(value));
      }
    };

  template <typename MapType>
  struct from_python_dict
  {
    typedef typename MapType::key_type k_t;
    typedef typename MapType::mapped_type m_t;

    from_python_dict()
    {
      boost::python::converter::registry::push_back(
        &convertible,
        &construct,
        boost::python::type_id<MapType>());
    }

    static void* convertible(PyObject* obj_ptr)
    {
      return PyDict_Check(obj_ptr) ? obj_ptr : 0;
    }

    static void construct(
      PyObject* obj_ptr,
      boost::python::converter::rvalue_from_python_stage1_data* data)
    {
      boost::python::handle<> obj_hdl(boost::python::borrowed(obj_ptr));
      boost::python::object obj_obj(obj_hdl);
      boost::python::extract<boost::python::dict> obj_proxy(obj_obj);
      boost::python::dict other = obj_proxy();
      void* storage = (
        (boost::python::converter::rvalue_from_python_storage<MapType>*)
          data)->storage.bytes;
      new (storage) MapType();
      data->convertible = storage;
      MapType& self = *((MapType*)storage);
      boost::python::list keys = other.keys();
      int len_keys = boost::python::len(keys);
      for(int i=0;i<len_keys;i++) {
        boost::python::object key_obj = keys[i];
        boost::python::extract<k_t> key_proxy(key_obj);
        if (!key_proxy.check()) {
          PYTHON_ERROR(KeyError, "unsuitable type");
        }
        boost::python::object value_obj = other[key_obj];
        boost::python::extract<m_t> value_proxy(value_obj);
        if (!value_proxy.check()) {
          PYTHON_ERROR(ValueError, "unsuitable value");
        }
        assign<m_t>()(self, key_proxy(), value_proxy());
      }
    }
  };

}} // namespace Torch::python

#endif /* TORCH_CORE_PYTHON_MAP_CONTAINER_H */
