/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Tue 15 Feb 22:48:32 2011 
 *
 * @brief Some sugar to inherit from ArraysetCodec and ArrayCodec from python!
 */

#include <boost/python.hpp>

#include "database/ArrayCodec.h"
#include "database/ArraysetCodec.h"

using namespace boost::python;
namespace db = Torch::database;

//boost::python scaffoldings for virtualization

class ArrayCodecWrapper: public db::ArrayCodec, public wrapper<db::ArrayCodec> {

  public:

    void peek(const std::string& filename, Torch::core::array::ElementType& eltype, size_t& ndim, size_t* shape) const {
      list lshape;
      this->get_override("peek")(filename, eltype, lshape); 
      ndim = len(lshape);
      for (size_t i=0; i<ndim; ++i) shape[i] = extract<size_t>(lshape[i])();
    }

    db::detail::InlinedArrayImpl load(const std::string& filename) const {
      return this->get_override("load")(filename);
    }

    void save (const std::string& filename, const db::detail::InlinedArrayImpl& data) const {
      this->get_override("save")(filename, data);
    }

    const std::string& name () const {
      return this->get_override("name")();
    }

    const std::vector<std::string>& extensions () const {
      tuple exts = this->get_override("extensions")();
      m_extensions.clear();
      for (Py_ssize_t i=0; i<len(exts); ++i) 
        m_extensions.push_back(extract<std::string>(exts[i])()); 
      return m_extensions;
    }

  private:

    mutable std::vector<std::string> m_extensions;

};

void bind_database_codec() {

  class_<ArrayCodecWrapper, boost::noncopyable>("ArrayCodec")
    .def("peek", pure_virtual(&db::ArrayCodec::peek))
    .def("load", pure_virtual(&db::ArrayCodec::load))
    .def("save", pure_virtual(&db::ArrayCodec::save))
    .def("name", pure_virtual(&db::ArrayCodec::name), return_internal_reference<>())
    .def("extensions", pure_virtual(&db::ArrayCodec::extensions), return_internal_reference<>())
    ;
}
