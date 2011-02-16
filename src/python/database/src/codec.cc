/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Tue 15 Feb 22:48:32 2011 
 *
 * @brief Some sugar to inherit from ArraysetCodec and ArrayCodec from python!
 */

#include <boost/python.hpp>
#include <boost/make_shared.hpp>

#include "database/ArrayCodec.h"
#include "database/ArraysetCodec.h"
#include "database/ArrayCodecRegistry.h"
#include "database/ArraysetCodecRegistry.h"

using namespace boost::python;
namespace db = Torch::database;
namespace array = Torch::core::array;

//boost::python scaffoldings for virtualization

/**
 * The ArrayCodecWrapper is a python reflection that allows users to implement
 * ArrayCodec's in pure python
 */
class ArrayCodecWrapper: public db::ArrayCodec, public wrapper<db::ArrayCodec> {

  public:

    /**
     * The peek() method that is overrideable in python will require your
     * method to receive a filename and return a tuple containing the element
     * type and a shape tuple.
     *
     * python prototype:
     *
     * def peek (filename):
     *   ...
     *   return (elementType, shape)
     *
     * shape = (extent0, extent1, ...)
     */
    void peek(const std::string& filename, array::ElementType& eltype,
        size_t& ndim, size_t* shape) const {
      tuple retval = this->get_override("peek")(filename);
      eltype = extract<array::ElementType>(retval[0])();
      ndim = len(retval[1]);
      for (size_t i=0; i<ndim; ++i) 
        shape[i] = extract<size_t>(retval[1][i])();
    }

    /**
     * The load() method takes a filename and returns a single db::Array (with
     * inlined representation)
     */
    db::detail::InlinedArrayImpl load(const std::string& filename) const {
      object retval = this->get_override("load")(filename);
      return extract<db::Array>(retval)().get();
    }

    void save (const std::string& filename, const db::detail::InlinedArrayImpl& data) const {
      this->get_override("save")(filename, db::Array(data));
    }

    const std::string& name () const {
      object o = this->get_override("name")();
      m_name = extract<std::string>(o)();
      return m_name;
    }

    const std::vector<std::string>& extensions () const {
      tuple exts = this->get_override("extensions")();
      m_extensions.clear();
      for (Py_ssize_t i=0; i<len(exts); ++i)
        m_extensions.push_back(extract<std::string>(exts[i])()); 
      return m_extensions;
    }

  private:

    mutable std::string m_name;
    mutable std::vector<std::string> m_extensions;

};

static tuple get_array_codec_names() {
  list retval;
  std::vector<std::string> names;
  db::ArrayCodecRegistry::getCodecNames(names);
  for (std::vector<std::string>::const_iterator it=names.begin(); it != names.end(); ++it) retval.append(*it);
  return tuple(retval);
}

static tuple get_array_extensions() {
  list retval;
  std::vector<std::string> names;
  db::ArrayCodecRegistry::getExtensions(names);
  for (std::vector<std::string>::const_iterator it=names.begin(); it != names.end(); ++it) retval.append(*it);
  return tuple(retval);
}

/**
 * Need to override the save method of the Cxx ArrayCodec as it normally does
 * not receive an Array as parameter, but an InlinedArrayImpl.
 */
static void cxx_array_codec_save (boost::shared_ptr<const db::ArrayCodec> codec, const std::string& filename, boost::shared_ptr<db::Array> array) {
  codec->save(filename, array->get());
}

static boost::shared_ptr<db::Array> cxx_array_codec_load (boost::shared_ptr<const db::ArrayCodec> codec, const std::string& filename) {
  return boost::make_shared<db::Array>(codec->load(filename));
}

void bind_database_codec() {
  class_<boost::shared_ptr<const db::ArrayCodec> >("CxxArrayCodec", no_init)
    .def("peek", &db::ArrayCodec::peek)
    .def("load", &cxx_array_codec_load)
    .def("save", &cxx_array_codec_save)
    .def("name", &db::ArrayCodec::name, return_internal_reference<>())
    .def("extensions", &db::ArrayCodec::extensions, return_internal_reference<>())
    ;

  class_<db::ArrayCodecRegistry, boost::shared_ptr<db::ArrayCodecRegistry>, boost::noncopyable>("ArrayCodecRegistry", "A Registry for Array Codecs available at runtime", no_init)
    .def("addCodec", &db::ArrayCodecRegistry::addCodec)
    .staticmethod("addCodec")
    .def("getCodecByName", &db::ArrayCodecRegistry::getCodecByName)
    .staticmethod("getCodecByName")
    .def("getCodecByExtension", &db::ArrayCodecRegistry::getCodecByExtension)
    .staticmethod("getCodecByExtension")
    .def("getCodecNames", &get_array_codec_names)
    .staticmethod("getCodecNames")
    .def("getExtensions", &get_array_extensions)
    .staticmethod("getExtensions")
    ;

  class_<ArrayCodecWrapper, boost::shared_ptr<ArrayCodecWrapper>, boost::noncopyable>("ArrayCodec")
    .def("peek", pure_virtual(&db::ArrayCodec::peek))
    .def("load", pure_virtual(&db::ArrayCodec::load))
    .def("save", pure_virtual(&db::ArrayCodec::save))
    .def("name", pure_virtual(&db::ArrayCodec::name), return_internal_reference<>())
    .def("extensions", pure_virtual(&db::ArrayCodec::extensions), return_internal_reference<>())
    ;
}
