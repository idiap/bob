/**
 * @file io/python/file.cc
 * @date Fri Oct 28 10:28:48 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Bindings for i/o files.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

#include <boost/python.hpp>
#include <bob/io/CodecRegistry.h>
#include <bob/io/File.h>
#include <bob/io/utils.h>

#include <bob/core/python/ndarray.h>

using namespace boost::python;

static object file_read_all(bob::io::File& f) {
  bob::python::py_array a(f.type_all());
  f.read_all(a);
  return a.pyobject(); //shallow copy
}

static object file_read(bob::io::File& f, size_t index) {
  bob::python::py_array a(f.type_all());
  f.read(a, index);
  return a.pyobject(); //shallow copy
}

static boost::shared_ptr<bob::io::File> string_open1 (const std::string& filename,
    const std::string& mode) {
  return bob::io::open(filename, mode[0]);
}

static boost::shared_ptr<bob::io::File> string_open2 (const std::string& filename,
    const std::string& mode, const std::string& pretend_extension) {
  return bob::io::open(filename, mode[0], pretend_extension);
}

static void file_write(bob::io::File& f, object array) {
  bob::python::py_array a(array, object());
  f.write(a);
}

static void file_append(bob::io::File& f, object array) {
  bob::python::py_array a(array, object());
  f.append(a);
}

static dict extensions() {
  typedef std::map<std::string, std::string> map_type;
  dict retval;
  const map_type& table = bob::io::CodecRegistry::getExtensions();
  for (map_type::const_iterator it=table.begin(); it!=table.end(); ++it) {
    retval[it->first] = it->second;
  }
  return retval;
}

void bind_io_file() {
  
  class_<bob::io::File, boost::shared_ptr<bob::io::File>, boost::noncopyable>("File", "Abstract base class for all Array/Arrayset i/o operations", no_init)
    .def("__init__", make_constructor(string_open1, default_call_policies(), (arg("filename"), arg("mode"))), "Opens a (supported) file for reading arrays. The mode is a **single** character which takes one of the following values: 'r' - opens the file for read-only operations; 'w' - truncates the file and open it for reading and writing; 'a' - opens the file for reading and writing w/o truncating it.")
    .def("__init__", make_constructor(string_open2, default_call_policies(), (arg("filename"), arg("mode"), arg("pretend_extension"))), "Opens a (supported) file for reading arrays but pretends its extension is as given by the last parameter - this way you can, potentially, override the default encoder/decoder used to read and write on the file. The mode is a **single** character which takes one of the following values: 'r' - opens the file for read-only operations; 'w' - truncates the file and open it for reading and writing; 'a' - opens the file for reading and writing w/o truncating it.")
    .add_property("filename", make_function(&bob::io::File::filename, return_value_policy<copy_const_reference>()), "The path to the file being read/written")
    .add_property("type_all", make_function(&bob::io::File::type_all, return_value_policy<copy_const_reference>()), "Typing information to load all of the file at once")
    .add_property("type", make_function(&bob::io::File::type, return_value_policy<copy_const_reference>()), "Typing information to load the file as an Arrayset")
    .add_property("codec_name", make_function(&bob::io::File::name, return_value_policy<copy_const_reference>()), "Name of the File class implementation -- for compatibility reasons with the previous versions of this library")
    .def("read", &file_read_all, (arg("self")), "Reads the whole contents of the file into a NumPy ndarray")
    .def("write", &file_write, (arg("self"), arg("array")), "Writes an array into the file, truncating it first")
    .def("__len__", &bob::io::File::size, (arg("self")), "Size of the file if it is supposed to be read as a set of arrays instead of performing a single read")
    .def("read", &file_read, (arg("self"), arg("index")), "Reads a single array from the file considering it to be an arrayset list")
    .def("__getitem__", &file_read, (arg("self"), arg("index")), "Reads a single array from the file considering it to be an arrayset list")
    .def("append", &file_append, (arg("self"), arg("array")), "Appends an array to a file. Compatibility requirements may be enforced.")
    ;

  def("extensions", &extensions, "Returns a dictionary containing all extensions and descriptions currently stored on the global codec registry");

}
