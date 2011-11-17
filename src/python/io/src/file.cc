/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu 27 Oct 15:15:57 2011
 *
 * @brief Bindings for i/o files.
 */

#include <boost/python.hpp>
#include "io/File.h"
#include "io/CodecRegistry.h"

#include "core/python/ndarray.h"

using namespace boost::python;
namespace tp = Torch::python;
namespace io = Torch::io;
namespace ca = Torch::core::array;

static object file_array_read(io::File& f) {
  tp::py_array a(f.array_type());
  f.array_read(a);
  return a.pyobject(); //shallow copy
}

static object file_arrayset_read(io::File& f, size_t index) {
  tp::py_array a(f.array_type());
  f.arrayset_read(a, index);
  return a.pyobject(); //shallow copy
}

static boost::shared_ptr<io::File> string_open1 (const std::string& filename,
    const std::string& mode) {
  return io::open(filename, "", mode[0]);
}

static boost::shared_ptr<io::File> string_open2 (const std::string& filename,
    const std::string& mode, const std::string& pretend_extension) {
  return io::open(filename, pretend_extension, mode[0]);
}

static void file_array_write(io::File& f, object array) {
  tp::py_array a(array, object());
  f.array_write(a);
}

static void file_arrayset_append(io::File& f, object array) {
  tp::py_array a(array, object());
  f.arrayset_append(a);
}

static dict extensions() {
  typedef std::map<std::string, std::string> map_type;
  dict retval;
  const map_type& table = io::CodecRegistry::getExtensions();
  for (map_type::const_iterator it=table.begin(); it!=table.end(); ++it) {
    retval[it->first] = it->second;
  }
  return retval;
}

void bind_io_file() {
  
  class_<io::File, boost::shared_ptr<io::File>, boost::noncopyable>("File", "Abstract base class for all Array/Arrayset i/o operations", no_init)
    .add_property("filename", make_function(&io::File::filename, return_value_policy<copy_const_reference>()), "The path to the file being read/written")
    .add_property("array_type", make_function(&io::File::array_type, return_value_policy<copy_const_reference>()), "Typing information to load all of the file at once")
    .add_property("arrayset_type", make_function(&io::File::array_type, return_value_policy<copy_const_reference>()), "Typing information to load the file as an Arrayset")
    .add_property("codec_name", make_function(&io::File::name, return_value_policy<copy_const_reference>()), "Name of the File class implementation -- for compatibility reasons with the previous versions of this library")
    .def("read", &file_array_read, (arg("self")), "Reads the whole contents of the file into a NumPy ndarray")
    .def("write", &file_array_write, (arg("self"), arg("array")), "Writes an array into the file, truncating it first")
    
    .def("__len__", &io::File::arrayset_size, (arg("self")), "Size of the file if it is supposed to be read as a set of arrays instead of performing a single read")
    .def("read", &file_arrayset_read, (arg("self"), arg("index")), "Reads a single array from the file considering it to be an arrayset list")
    .def("__getitem__", &file_arrayset_read, (arg("self"), arg("index")), "Reads a single array from the file considering it to be an arrayset list")
    .def("append", &file_arrayset_append, (arg("self"), arg("array")), "Appends an array to a file. Compatibility requirements may be enforced.")
    ;

  def("open", &string_open1, (arg("filename"), arg("mode")), "Opens a (supported) file for reading arrays. The mode is a **single** character which takes one of the following values: 'r' - opens the file for read-only operations; 'w' - truncates the file and open it for reading and writing; 'a' - opens the file for reading and writing w/o truncating it.");
  def("open", &string_open2, (arg("filename"), arg("mode"), arg("pretend_extension")), "Opens a (supported) file for reading arrays but pretends its extension is as given by the last parameter - this way you can, potentially, override the default encoder/decoder used to read and write on the file. The mode is a **single** character which takes one of the following values: 'r' - opens the file for read-only operations; 'w' - truncates the file and open it for reading and writing; 'a' - opens the file for reading and writing w/o truncating it.");

  def("extensions", &extensions, "Returns a dictionary containing all extensions and descriptions currently stored on the global codec registry");

}
