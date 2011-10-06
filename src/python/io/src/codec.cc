/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
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

#include "core/python/pycore.h"

using namespace boost::python;
namespace io = Torch::io;
namespace array = Torch::core::array;
namespace tp = Torch::python;

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

static tuple cxx_array_codec_peek (const io::ArrayCodec& codec, const std::string& filename) {
  array::ElementType eltype;
  size_t ndim;
  size_t shape[array::N_MAX_DIMENSIONS_ARRAY];
  codec->peek(filename, eltype, ndim, shape);
  list lshape;
  for (size_t i=0; i<ndim; ++i) lshape.append(shape[i]);
  return make_tuple(array::stringize(eltype), tuple(lshape));
}

static void cxx_array_codec_save (const io::ArrayCodec& codec, 
    const std::string& filename, numeric::array& arr) {

  codec->save(filename, array->get());
}

static io::Array& cxx_array_codec_load (const io::ArrayCodec& codec, const std::string& filename) {
  return boost::make_shared<io::Array>(codec->load(filename));
}

static const char* cxx_array_codec_name (const io::ArrayCodec& codec) {
  return codec->name().c_str();
}

static tuple cxx_array_extensions (const io::ArrayCodec& codec) {
  list retval;
  for (std::vector<std::string>::const_iterator it=codec->extensions().begin(); it != codec->extensions().end(); ++it) retval.append(*it);
  return tuple(retval);
}

/**
 * Need to override methods of the Cxx ArraysetCodec to make it match the
 * pythonic extensions
 */
static tuple cxx_arrayset_codec_peek (const io::ArraysetCodec& codec, 
    const std::string& filename) {
  array::ElementType eltype;
  size_t ndim;
  size_t shape[array::N_MAX_DIMENSIONS_ARRAY];
  size_t samples;
  codec->peek(filename, eltype, ndim, shape, samples);
  list lshape;
  for (size_t i=0; i<ndim; ++i) lshape.append(shape[i]);
  return make_tuple(array::stringize(eltype), tuple(lshape), samples);
}

static io::Arrayset& cxx_arrayset_codec_load (const io::ArraysetCodec& codec, const std::string& filename) {
  return boost::make_shared<io::Arrayset>(codec->load(filename));
}

static io::Array& cxx_arrayset_codec_load_one (const io::ArraysetCodec& codec, const std::string& filename, size_t index) {
  return boost::make_shared<io::Array>(codec->load(filename, index));
}

static void cxx_arrayset_codec_save (const io::ArraysetCodec& codec, const std::string& filename, io::Arrayset& arrayset) {
  codec->save(filename, arrayset->get());
}

static void cxx_arrayset_codec_append (const io::ArraysetCodec& codec, const std::string& filename, io::Array& array) {
  codec->append(filename, *array.get());
}

static tuple cxx_arrayset_extensions (const io::ArraysetCodec& codec) {
  list retval;
  for (std::vector<std::string>::const_iterator it=codec->extensions().begin(); it != codec->extensions().end(); ++it) retval.append(*it);
  return tuple(retval);
}

static const char* cxx_arrayset_codec_name (const io::ArraysetCodec& codec) {
  return codec->name().c_str();
}

void bind_io_codec() {
  class_<io::ArrayCodec, boost::noncopyable>("ArrayCodec", no_init)
    .def("peek", &cxx_array_codec_peek)
    .def("load", &cxx_array_codec_load)
    .def("save", &cxx_array_codec_save)
    .def("name", &cxx_array_codec_name)
    .def("extensions", &cxx_array_extensions)
    ;

  class_<io::ArraysetCodec, boost::noncopyable>("ArraysetCodec", no_init)
    .def("peek", &cxx_arrayset_codec_peek)
    .def("load", &cxx_arrayset_codec_load)
    .def("load", &cxx_arrayset_codec_load_one)
    .def("save", &cxx_arrayset_codec_save)
    .def("append", &cxx_arrayset_codec_append)
    .def("name", &cxx_arrayset_codec_name)
    .def("extensions", &cxx_arrayset_extensions)
    ;

  class_<io::ArrayCodecRegistry, boost::noncopyable>("ArrayCodecRegistry", "A Registry for Array Codecs available at runtime", no_init)
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

  class_<io::ArraysetCodecRegistry, boost::noncopyable>("ArraysetCodecRegistry", "A Registry for Arrayset Codecs available at runtime", no_init)
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
}
