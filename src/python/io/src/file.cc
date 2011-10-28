/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu 27 Oct 15:15:57 2011
 *
 * @brief Bindings for i/o files.
 */

#include <boost/python.hpp>
#include "io/File.h"
#include "io/CodecRegistry.h"

#include "io/python/pyio.h"

using namespace boost::python;
namespace tp = Torch::python;
namespace io = Torch::io;

static tuple ti_shape(const io::typeinfo& ti) {
  list retval;
  for (size_t i=0; i<ti.nd; ++i) retval.append(ti.shape[i]);
  return tuple(retval);
}

static tuple ti_stride(const io::typeinfo& ti) {
  list retval;
  for (size_t i=0; i<ti.nd; ++i) retval.append(ti.stride[i]);
  return tuple(retval);
}

static object file_array_read(io::File& f) {
  tp::npyarray a(f.array_type());
  f.array_read(a);
  return tp::npyarray_object(a);
}

static object file_arrayset_read(io::File& f, size_t index) {
  tp::npyarray a(f.array_type());
  f.arrayset_read(a, index);
  return tp::npyarray_object(a);
}

static boost::shared_ptr<io::File> string_open1 (const std::string& filename,
    const std::string& mode) {
  return io::open(filename, "", mode[0]);
}

static boost::shared_ptr<io::File> string_open2 (const std::string& filename,
    const std::string& mode, const std::string& pretend_extension) {
  return io::open(filename, pretend_extension, mode[0]);
}

void bind_io_file() {
  
  class_<io::typeinfo>("typeinfo", "Type information for Torch C++ data", 
      no_init)
    .def_readonly("dtype", &io::typeinfo::dtype)
    .def_readonly("nd", &io::typeinfo::nd)
    .add_property("shape", &ti_shape)
    .add_property("stride", &ti_stride)
    ;

  class_<io::File, boost::shared_ptr<io::File>, boost::noncopyable>("File", "Abstract base class for all Array/Arrayset i/o operations", no_init)
    .add_property("filename", make_function(&io::File::filename, return_value_policy<copy_const_reference>()), "The path to the file being read/written")
    .add_property("array_type", make_function(&io::File::array_type, return_value_policy<copy_const_reference>()), "Typing information to load all of the file at once")
    .add_property("arrayset_type", make_function(&io::File::array_type, return_value_policy<copy_const_reference>()), "Typing information to load the file as an Arrayset")
    .add_property("codec_name", make_function(&io::File::name, return_value_policy<copy_const_reference>()), "Name of the File class implementation -- for compatibility reasons with the previous versions of this library")
    .def("read", &file_array_read, (arg("self")), "Reads the whole contents of the file into a NumPy ndarray")
    .def("write", &io::File::array_write, (arg("self"), arg("array")), "Writes an array into the file, truncating it first")
    
    .def("__len__", &io::File::arrayset_size, (arg("self")), "Size of the file if it is supposed to be read as a set of arrays instead of performing a single read")
    .def("read", &file_arrayset_read, (arg("self"), arg("index")), "Reads a single array from the file considering it to be an arrayset list")
    .def("__getitem__", &file_arrayset_read, (arg("self"), arg("index")), "Reads a single array from the file considering it to be an arrayset list")
    .def("append", &io::File::arrayset_append, (arg("self"), arg("array")), "Appends an array to a file. Compatibility requirements may be enforced.")
    ;

  def("open", &string_open1, (arg("filename"), arg("mode")), "Opens a (supported) file for reading arrays. The mode is a **single** character which takes one of the following values: 'r' - opens the file for read-only operations; 'w' - truncates the file and open it for reading and writing; 'a' - opens the file for reading and writing w/o truncating it.");
  def("open", &string_open2, (arg("filename"), arg("mode"), arg("pretend_extension")), "Opens a (supported) file for reading arrays but pretends its extension is as given by the last parameter - this way you can, potentially, override the default encoder/decoder used to read and write on the file. The mode is a **single** character which takes one of the following values: 'r' - opens the file for read-only operations; 'w' - truncates the file and open it for reading and writing; 'a' - opens the file for reading and writing w/o truncating it.");

}
