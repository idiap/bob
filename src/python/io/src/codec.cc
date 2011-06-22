/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Tue 15 Feb 22:48:32 2011 
 *
 * @brief Some sugar to inherit from ArraysetCodec and ArrayCodec from python!
 */

#include <boost/python.hpp>
#include <boost/make_shared.hpp>

#include "io/ArrayCodec.h"
#include "io/ArraysetCodec.h"
#include "io/ArrayCodecRegistry.h"
#include "io/ArraysetCodecRegistry.h"

#include "io/Array.h"
#include "io/Arrayset.h"

using namespace boost::python;
namespace io = Torch::io;
namespace array = Torch::core::array;

//boost::python scaffoldings for virtualization

/**
 * The ArrayCodecWrapper is a python reflection that allows users to implement
 * ArrayCodec's in pure python
 */
class ArrayCodecWrapper: public io::ArrayCodec, public wrapper<io::ArrayCodec> {

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
     * The load() method takes a filename and returns a single io::Array (with
     * inlined representation)
     */
    io::detail::InlinedArrayImpl load(const std::string& filename) const {
      object retval = this->get_override("load")(filename);
      return extract<io::Array>(retval)().get();
    }

    void save (const std::string& filename, const io::detail::InlinedArrayImpl& data) const {
      this->get_override("save")(filename, io::Array(data));
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
class ArraysetCodecWrapper: public io::ArraysetCodec, public wrapper<io::ArraysetCodec> {

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
    io::detail::InlinedArraysetImpl load(const std::string& filename) const {
      object retval = this->get_override("load")(filename);
      return extract<io::Arrayset>(retval)().get();
    }

    /**
     * The load() method takes a filename and a relative index position and
     * returns a single io::Array (with inlined representation). The first
     * position in the file is addressed as "1" (fortran-based counting).
     */
    io::Array load(const std::string& filename, size_t index) const {
      object retval = this->get_override("load")(filename, index);
      return extract<io::Array>(retval)();
    }

    /**
     * Append adds a single Array to the pool
     */
    void append (const std::string& filename, const io::Array& data) const {
      this->get_override("append")(filename, io::Array(data));
    }

    void save (const std::string& filename, const io::detail::InlinedArraysetImpl& data) const {
      this->get_override("save")(filename, io::Arrayset(data));
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
  io::ArrayCodecRegistry::getCodecNames(names);
  for (std::vector<std::string>::const_iterator it=names.begin(); it != names.end(); ++it) retval.append(*it);
  return tuple(retval);
}

static tuple get_array_extensions() {
  list retval;
  std::vector<std::string> names;
  io::ArrayCodecRegistry::getExtensions(names);
  for (std::vector<std::string>::const_iterator it=names.begin(); it != names.end(); ++it) retval.append(*it);
  return tuple(retval);
}

static tuple get_arrayset_codec_names() {
  list retval;
  std::vector<std::string> names;
  io::ArraysetCodecRegistry::getCodecNames(names);
  for (std::vector<std::string>::const_iterator it=names.begin(); it != names.end(); ++it) retval.append(*it);
  return tuple(retval);
}

static tuple get_arrayset_extensions() {
  list retval;
  std::vector<std::string> names;
  io::ArraysetCodecRegistry::getExtensions(names);
  for (std::vector<std::string>::const_iterator it=names.begin(); it != names.end(); ++it) retval.append(*it);
  return tuple(retval);
}

/**
 * Need to override methods of the Cxx ArrayCodec to make it match the pythonic
 * extensions
 */
static tuple cxx_array_codec_peek (boost::shared_ptr<const io::ArrayCodec> codec, const std::string& filename) {
  array::ElementType eltype;
  size_t ndim;
  size_t shape[array::N_MAX_DIMENSIONS_ARRAY];
  codec->peek(filename, eltype, ndim, shape);
  list lshape;
  for (size_t i=0; i<ndim; ++i) lshape.append(shape[i]);
  return make_tuple(eltype, tuple(lshape));
}

static void cxx_array_codec_save (boost::shared_ptr<const io::ArrayCodec> codec, const std::string& filename, boost::shared_ptr<io::Array> array) {
  codec->save(filename, array->get());
}

static boost::shared_ptr<io::Array> cxx_array_codec_load (boost::shared_ptr<const io::ArrayCodec> codec, const std::string& filename) {
  return boost::make_shared<io::Array>(codec->load(filename));
}

static const char* cxx_array_codec_name (boost::shared_ptr<const io::ArrayCodec> codec) {
  return codec->name().c_str();
}

static tuple cxx_array_extensions (boost::shared_ptr<const io::ArrayCodec> codec) {
  list retval;
  for (std::vector<std::string>::const_iterator it=codec->extensions().begin(); it != codec->extensions().end(); ++it) retval.append(*it);
  return tuple(retval);
}

/**
 * Need to override methods of the Cxx ArraysetCodec to make it match the
 * pythonic extensions
 */
static tuple cxx_arrayset_codec_peek (boost::shared_ptr<const io::ArraysetCodec> codec, const std::string& filename) {
  array::ElementType eltype;
  size_t ndim;
  size_t shape[array::N_MAX_DIMENSIONS_ARRAY];
  size_t samples;
  codec->peek(filename, eltype, ndim, shape, samples);
  list lshape;
  for (size_t i=0; i<ndim; ++i) lshape.append(shape[i]);
  return make_tuple(eltype, tuple(lshape), samples);
}

static boost::shared_ptr<io::Arrayset> cxx_arrayset_codec_load (boost::shared_ptr<const io::ArraysetCodec> codec, const std::string& filename) {
  return boost::make_shared<io::Arrayset>(codec->load(filename));
}

static boost::shared_ptr<io::Array> cxx_arrayset_codec_load_one (boost::shared_ptr<const io::ArraysetCodec> codec, const std::string& filename, size_t index) {
  return boost::make_shared<io::Array>(codec->load(filename, index));
}

static void cxx_arrayset_codec_save (boost::shared_ptr<const io::ArraysetCodec> codec, const std::string& filename, boost::shared_ptr<io::Arrayset> arrayset) {
  codec->save(filename, arrayset->get());
}

static void cxx_arrayset_codec_append (boost::shared_ptr<const io::ArraysetCodec> codec, const std::string& filename, boost::shared_ptr<io::Array> array) {
  codec->append(filename, *array.get());
}

static tuple cxx_arrayset_extensions (boost::shared_ptr<const io::ArraysetCodec> codec) {
  list retval;
  for (std::vector<std::string>::const_iterator it=codec->extensions().begin(); it != codec->extensions().end(); ++it) retval.append(*it);
  return tuple(retval);
}

static const char* cxx_arrayset_codec_name (boost::shared_ptr<const io::ArraysetCodec> codec) {
  return codec->name().c_str();
}

void bind_io_codec() {
  class_<boost::shared_ptr<const io::ArrayCodec> >("CxxArrayCodec", no_init)
    .def("peek", &cxx_array_codec_peek)
    .def("load", &cxx_array_codec_load)
    .def("save", &cxx_array_codec_save)
    .def("name", &cxx_array_codec_name)
    .def("extensions", &cxx_array_extensions)
    ;

  class_<boost::shared_ptr<const io::ArraysetCodec> >("CxxArraysetCodec", no_init)
    .def("peek", &cxx_arrayset_codec_peek)
    .def("load", &cxx_arrayset_codec_load)
    .def("load", &cxx_arrayset_codec_load_one)
    .def("save", &cxx_arrayset_codec_save)
    .def("append", &cxx_arrayset_codec_append)
    .def("name", &cxx_arrayset_codec_name)
    .def("extensions", &cxx_arrayset_extensions)
    ;

  class_<io::ArrayCodecRegistry, boost::shared_ptr<io::ArrayCodecRegistry>, boost::noncopyable>("ArrayCodecRegistry", "A Registry for Array Codecs available at runtime", no_init)
    .def("addCodec", &io::ArrayCodecRegistry::addCodec)
    .staticmethod("addCodec")
    .def("removeCodecByName", &io::ArrayCodecRegistry::removeCodecByName)
    .staticmethod("removeCodecByName")
    .def("getCodecByName", &io::ArrayCodecRegistry::getCodecByName)
    .staticmethod("getCodecByName")
    .def("getCodecByExtension", &io::ArrayCodecRegistry::getCodecByExtension)
    .staticmethod("getCodecByExtension")
    .def("getCodecNames", &get_array_codec_names)
    .staticmethod("getCodecNames")
    .def("getExtensions", &get_array_extensions)
    .staticmethod("getExtensions")
    ;

  class_<io::ArraysetCodecRegistry, boost::shared_ptr<io::ArraysetCodecRegistry>, boost::noncopyable>("ArraysetCodecRegistry", "A Registry for Arrayset Codecs available at runtime", no_init)
    .def("addCodec", &io::ArraysetCodecRegistry::addCodec)
    .staticmethod("addCodec")
    .def("removeCodecByName", &io::ArraysetCodecRegistry::removeCodecByName)
    .staticmethod("removeCodecByName")
    .def("getCodecByName", &io::ArraysetCodecRegistry::getCodecByName)
    .staticmethod("getCodecByName")
    .def("getCodecByExtension", &io::ArraysetCodecRegistry::getCodecByExtension)
    .staticmethod("getCodecByExtension")
    .def("getCodecNames", &get_arrayset_codec_names)
    .staticmethod("getCodecNames")
    .def("getExtensions", &get_arrayset_extensions)
    .staticmethod("getExtensions")
    ;

  class_<ArrayCodecWrapper, boost::shared_ptr<ArrayCodecWrapper>, boost::noncopyable>("ArrayCodec")
    .def("peek", pure_virtual(&io::ArrayCodec::peek))
    .def("load", pure_virtual(&io::ArrayCodec::load))
    .def("save", pure_virtual(&io::ArrayCodec::save))
    .def("name", pure_virtual(&io::ArrayCodec::name), return_internal_reference<>())
    .def("extensions", pure_virtual(&io::ArrayCodec::extensions), return_internal_reference<>())
    ;

  class_<ArraysetCodecWrapper, boost::shared_ptr<ArraysetCodecWrapper>, boost::noncopyable>("ArraysetCodec")
    .def("peek", pure_virtual(&io::ArraysetCodec::peek))
    .def("load", pure_virtual((io::detail::InlinedArraysetImpl (io::ArraysetCodec::*)(const std::string&) const)&io::ArraysetCodec::load))
    .def("load", pure_virtual((io::Array (io::ArraysetCodec::*)(const std::string&, size_t) const)&io::ArraysetCodec::load))
    .def("save", pure_virtual(&io::ArraysetCodec::save))
    .def("name", pure_virtual(&io::ArraysetCodec::name), return_internal_reference<>())
    .def("extensions", pure_virtual(&io::ArraysetCodec::extensions), return_internal_reference<>())
    ;
}
