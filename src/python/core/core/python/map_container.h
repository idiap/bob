/**
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch> 
 * @author Andre Anjos <andre.anjos@idiap.ch> 
 * @date Tue 22 Jul 2011 15:55:25 CEST 
 *
 * @brief Generic container conversions for boost::python, extracted from: http://cctbx.cvs.sourceforge.net/viewvc/cctbx/scitbx/include/scitbx/stl/map_wrapper.h
 *
 * version 1.16 (not touched for at least 5 years).
 **/

#ifndef TORCH_CORE_PYTHON_MAP_CONTAINER_H 
#define TORCH_CORE_PYTHON_MAP_CONTAINER_H

#include <boost/python.hpp>

namespace Torch { namespace python {

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
          PyErr_SetString(PyExc_KeyError, "Unsuitable type.");
          boost::python::throw_error_already_set();
        }
        boost::python::object value_obj = other[key_obj];
        boost::python::extract<m_t> value_proxy(value_obj);
        if (!value_proxy.check()) {
          PyErr_SetString(PyExc_ValueError, "Unsuitable type.");
          boost::python::throw_error_already_set();
        }
        k_t key = key_proxy();
        m_t value = value_proxy();
        self[key] = value;
      }
    }
  };

}} // namespace Torch::python

#endif /* TORCH_CORE_PYTHON_MAP_CONTAINER_H */
