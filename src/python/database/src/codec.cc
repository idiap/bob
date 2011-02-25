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

#include "database/Array.h"
#include "database/Arrayset.h"

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

/**
 * The ArraysetCodecWrapper is a python reflection that allows users to
 * implement ArraysetCodec's in pure python
 */
class ArraysetCodecWrapper: public db::ArraysetCodec, public wrapper<db::ArraysetCodec> {

  public:

    /**
     * The peek() method that is overrideable in python will require your
     * method to receive a filename and return a tuple containing the element
     * type and a shape tuple and number of samples in the file.
     *
     * python prototype:
     *
     * def peek (filename):
     *   ...
     *   return (elementType, shape, samples)
     *
     * shape = (extent0, extent1, ...)
     */
    void peek(const std::string& filename, array::ElementType& eltype,
        size_t& ndim, size_t* shape, size_t& samples) const {
      tuple retval = this->get_override("peek")(filename);
      eltype = extract<array::ElementType>(retval[0])();
      ndim = len(retval[1]);
      samples = extract<size_t>(retval[2])();
      for (size_t i=0; i<ndim; ++i) 
        shape[i] = extract<size_t>(retval[1][i])();
    }

    //loads the full array in one shot.
    db::detail::InlinedArraysetImpl load(const std::string& filename) const {
      object retval = this->get_override("load")(filename);
      return extract<db::Arrayset>(retval)().get();
    }

    /**
     * The load() method takes a filename and a relative index position and
     * returns a single db::Array (with inlined representation). The first
     * position in the file is addressed as "1" (fortran-based counting).
     */
    db::Array load(const std::string& filename, size_t index) const {
      object retval = this->get_override("load")(filename, index);
      return extract<db::Array>(retval)();
    }

    /**
     * Append adds a single Array to the pool
     */
    void append (const std::string& filename, const db::Array& data) const {
      this->get_override("append")(filename, db::Array(data));
    }

    void save (const std::string& filename, const db::detail::InlinedArraysetImpl& data) const {
      this->get_override("save")(filename, db::Arrayset(data));
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

static tuple get_arrayset_codec_names() {
  list retval;
  std::vector<std::string> names;
  db::ArraysetCodecRegistry::getCodecNames(names);
  for (std::vector<std::string>::const_iterator it=names.begin(); it != names.end(); ++it) retval.append(*it);
  return tuple(retval);
}

static tuple get_arrayset_extensions() {
  list retval;
  std::vector<std::string> names;
  db::ArraysetCodecRegistry::getExtensions(names);
  for (std::vector<std::string>::const_iterator it=names.begin(); it != names.end(); ++it) retval.append(*it);
  return tuple(retval);
}

/**
 * Need to override methods of the Cxx ArrayCodec to make it match the pythonic
 * extensions
 */
static tuple cxx_array_codec_peek (boost::shared_ptr<const db::ArrayCodec> codec, const std::string& filename) {
  array::ElementType eltype;
  size_t ndim;
  size_t shape[array::N_MAX_DIMENSIONS_ARRAY];
  codec->peek(filename, eltype, ndim, shape);
  list lshape;
  for (size_t i=0; i<ndim; ++i) lshape.append(shape[i]);
  return make_tuple(eltype, tuple(lshape));
}

static void cxx_array_codec_save (boost::shared_ptr<const db::ArrayCodec> codec, const std::string& filename, boost::shared_ptr<db::Array> array) {
  codec->save(filename, array->get());
}

static boost::shared_ptr<db::Array> cxx_array_codec_load (boost::shared_ptr<const db::ArrayCodec> codec, const std::string& filename) {
  return boost::make_shared<db::Array>(codec->load(filename));
}

static const char* cxx_array_codec_name (boost::shared_ptr<const db::ArrayCodec> codec) {
  return codec->name().c_str();
}

static tuple cxx_array_extensions (boost::shared_ptr<const db::ArrayCodec> codec) {
  list retval;
  for (std::vector<std::string>::const_iterator it=codec->extensions().begin(); it != codec->extensions().end(); ++it) retval.append(*it);
  return tuple(retval);
}

/**
 * Need to override methods of the Cxx ArraysetCodec to make it match the
 * pythonic extensions
 */
static tuple cxx_arrayset_codec_peek (boost::shared_ptr<const db::ArraysetCodec> codec, const std::string& filename) {
  array::ElementType eltype;
  size_t ndim;
  size_t shape[array::N_MAX_DIMENSIONS_ARRAY];
  size_t samples;
  codec->peek(filename, eltype, ndim, shape, samples);
  list lshape;
  for (size_t i=0; i<ndim; ++i) lshape.append(shape[i]);
  return make_tuple(eltype, tuple(lshape), samples);
}

static boost::shared_ptr<db::Arrayset> cxx_arrayset_codec_load (boost::shared_ptr<const db::ArraysetCodec> codec, const std::string& filename) {
  return boost::make_shared<db::Arrayset>(codec->load(filename));
}

static boost::shared_ptr<db::Array> cxx_arrayset_codec_load_one (boost::shared_ptr<const db::ArraysetCodec> codec, const std::string& filename, size_t index) {
  return boost::make_shared<db::Array>(codec->load(filename, index));
}

static void cxx_arrayset_codec_save (boost::shared_ptr<const db::ArraysetCodec> codec, const std::string& filename, boost::shared_ptr<db::Arrayset> arrayset) {
  codec->save(filename, arrayset->get());
}

static void cxx_arrayset_codec_append (boost::shared_ptr<const db::ArraysetCodec> codec, const std::string& filename, boost::shared_ptr<db::Array> array) {
  codec->append(filename, *array.get());
}

static tuple cxx_arrayset_extensions (boost::shared_ptr<const db::ArraysetCodec> codec) {
  list retval;
  for (std::vector<std::string>::const_iterator it=codec->extensions().begin(); it != codec->extensions().end(); ++it) retval.append(*it);
  return tuple(retval);
}

static const char* cxx_arrayset_codec_name (boost::shared_ptr<const db::ArraysetCodec> codec) {
  return codec->name().c_str();
}

void bind_database_codec() {
  class_<boost::shared_ptr<const db::ArrayCodec> >("CxxArrayCodec", no_init)
    .def("peek", &cxx_array_codec_peek)
    .def("load", &cxx_array_codec_load)
    .def("save", &cxx_array_codec_save)
    .def("name", &cxx_array_codec_name)
    .def("extensions", &cxx_array_extensions)
    ;

  class_<boost::shared_ptr<const db::ArraysetCodec> >("CxxArraysetCodec", no_init)
    .def("peek", &cxx_arrayset_codec_peek)
    .def("load", &cxx_arrayset_codec_load)
    .def("load", &cxx_arrayset_codec_load_one)
    .def("save", &cxx_arrayset_codec_save)
    .def("append", &cxx_arrayset_codec_append)
    .def("name", &cxx_arrayset_codec_name)
    .def("extensions", &cxx_arrayset_extensions)
    ;

  class_<db::ArrayCodecRegistry, boost::shared_ptr<db::ArrayCodecRegistry>, boost::noncopyable>("ArrayCodecRegistry", "A Registry for Array Codecs available at runtime", no_init)
    .def("addCodec", &db::ArrayCodecRegistry::addCodec)
    .staticmethod("addCodec")
    .def("removeCodecByName", &db::ArrayCodecRegistry::removeCodecByName)
    .staticmethod("removeCodecByName")
    .def("getCodecByName", &db::ArrayCodecRegistry::getCodecByName)
    .staticmethod("getCodecByName")
    .def("getCodecByExtension", &db::ArrayCodecRegistry::getCodecByExtension)
    .staticmethod("getCodecByExtension")
    .def("getCodecNames", &get_array_codec_names)
    .staticmethod("getCodecNames")
    .def("getExtensions", &get_array_extensions)
    .staticmethod("getExtensions")
    ;

  class_<db::ArraysetCodecRegistry, boost::shared_ptr<db::ArraysetCodecRegistry>, boost::noncopyable>("ArraysetCodecRegistry", "A Registry for Arrayset Codecs available at runtime", no_init)
    .def("addCodec", &db::ArraysetCodecRegistry::addCodec)
    .staticmethod("addCodec")
    .def("removeCodecByName", &db::ArraysetCodecRegistry::removeCodecByName)
    .staticmethod("removeCodecByName")
    .def("getCodecByName", &db::ArraysetCodecRegistry::getCodecByName)
    .staticmethod("getCodecByName")
    .def("getCodecByExtension", &db::ArraysetCodecRegistry::getCodecByExtension)
    .staticmethod("getCodecByExtension")
    .def("getCodecNames", &get_arrayset_codec_names)
    .staticmethod("getCodecNames")
    .def("getExtensions", &get_arrayset_extensions)
    .staticmethod("getExtensions")
    ;

  class_<ArrayCodecWrapper, boost::shared_ptr<ArrayCodecWrapper>, boost::noncopyable>("ArrayCodec")
    .def("peek", pure_virtual(&db::ArrayCodec::peek))
    .def("load", pure_virtual(&db::ArrayCodec::load))
    .def("save", pure_virtual(&db::ArrayCodec::save))
    .def("name", pure_virtual(&db::ArrayCodec::name), return_internal_reference<>())
    .def("extensions", pure_virtual(&db::ArrayCodec::extensions), return_internal_reference<>())
    ;

  class_<ArraysetCodecWrapper, boost::shared_ptr<ArraysetCodecWrapper>, boost::noncopyable>("ArraysetCodec")
    .def("peek", pure_virtual(&db::ArraysetCodec::peek))
    .def("load", pure_virtual((db::detail::InlinedArraysetImpl (db::ArraysetCodec::*)(const std::string&) const)&db::ArraysetCodec::load))
    .def("load", pure_virtual((db::Array (db::ArraysetCodec::*)(const std::string&, size_t) const)&db::ArraysetCodec::load))
    .def("save", pure_virtual(&db::ArraysetCodec::save))
    .def("name", pure_virtual(&db::ArraysetCodec::name), return_internal_reference<>())
    .def("extensions", pure_virtual(&db::ArraysetCodec::extensions), return_internal_reference<>())
    ;
}
