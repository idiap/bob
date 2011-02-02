/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 * @date Wed 26 Jan 2011 07:46:17
 *
 * @brief Python bindings to Torch::core::BinFile
 */

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>

#include "core/BinFile.h"

#include <iostream>

using namespace boost::python;
namespace db = Torch::core;

template <typename T>
static tuple get_shape(const T& f) {
  size_t ndim = f.getNDimensions();
  const size_t* shape = f.getShape();
  switch (ndim) {
    case 1:
      return make_tuple(shape[0]);
    case 2:
      return make_tuple(shape[0], shape[1]);
    case 3:
      return make_tuple(shape[0], shape[1], shape[2]);
    case 4:
      return make_tuple(shape[0], shape[1], shape[2], shape[3]);
    default:
      break;
  }
  return make_tuple();
}

/**
 * Converts an image from any format into grayscale.
 */
static boost::shared_ptr<Torch::core::BinFile> 
binfile_make_fromint(const std::string& filename, int i)
{
  boost::shared_ptr<Torch::core::BinFile> retval(new Torch::core::BinFile(
    filename, static_cast<Torch::core::BinFile::openmode>(i) ) );
  return retval;
}


static const char* ARRAY_READ_DOC = "Adapts the size of each dimension of the passed blitz array to the ones of the binary file and copies the data for a certain array in the file to the passed blitz::Array.";
static const char* ARRAY_WRITE_DOC = "Writes a single blitz::Array<> into the binary file. Please note that this array should conform to the shape and element type of the arrays already inserted. If no array was inserted, the element type and shape will be defined when you first write an array to this binary file.";
#define ARRAY_DEF(T,D) .def("bzread", (void (db::BinFile::*)(size_t, blitz::Array<T,D>&))&db::BinFile::read<T,D>, (arg("self"), arg("index"), arg("array")), ARRAY_READ_DOC) \
.def("append", (void (db::BinFile::*)(const blitz::Array<T,D>&))&db::BinFile::write<T,D>, (arg("self"), arg("array")), ARRAY_WRITE_DOC)


void bind_database_binfile() {
  enum_<Torch::core::BinFile::openmode>("openmode")
        .value("inp", Torch::core::BinFile::in)
        .value("out", Torch::core::BinFile::out)
        .value("append", Torch::core::BinFile::append)
        ;

  class_<db::BinFile, boost::shared_ptr<db::BinFile>, boost::noncopyable>("BinFile", "A BinFile allows users to read and write data from and to files containing standard Torch binary coded data", init<const std::string&, Torch::core::BinFile::openmode>((arg("filename"),arg("openmode")), "Initializes an binary file reader. Please note that this constructor will not load the data."))
    .def("__init__", make_constructor(binfile_make_fromint))
    .add_property("shape", &get_shape<db::BinFile>, "The shape of arrays in this binary file. Please note all arrays in the file have necessarily the same shape.")
    .add_property("elementType", &db::BinFile::getElementType, "The type of array elements contained in this binary file. This would be equivalent to the 'T' bit in blitz::Array<T,D>.")
    .def("__len__", &db::BinFile::getNSamples, "The number of arrays in this binary file.")
    ARRAY_DEF(bool, 1)
    ARRAY_DEF(int8_t, 1)
    ARRAY_DEF(int16_t, 1)
    ARRAY_DEF(int32_t, 1)
    ARRAY_DEF(int64_t, 1)
    ARRAY_DEF(uint8_t, 1)
    ARRAY_DEF(uint16_t, 1)
    ARRAY_DEF(uint32_t, 1)
    ARRAY_DEF(uint64_t, 1)
    ARRAY_DEF(float, 1)
    ARRAY_DEF(double, 1)
    ARRAY_DEF(std::complex<float>, 1)
    ARRAY_DEF(std::complex<double>, 1)
    ARRAY_DEF(bool, 2)
    ARRAY_DEF(int8_t, 2)
    ARRAY_DEF(int16_t, 2)
    ARRAY_DEF(int32_t, 2)
    ARRAY_DEF(int64_t, 2)
    ARRAY_DEF(uint8_t, 2)
    ARRAY_DEF(uint16_t, 2)
    ARRAY_DEF(uint32_t, 2)
    ARRAY_DEF(uint64_t, 2)
    ARRAY_DEF(float, 2)
    ARRAY_DEF(double, 2)
    ARRAY_DEF(std::complex<float>, 2)
    ARRAY_DEF(std::complex<double>, 2)
    ARRAY_DEF(bool, 3)
    ARRAY_DEF(int8_t, 3)
    ARRAY_DEF(int16_t, 3)
    ARRAY_DEF(int32_t, 3)
    ARRAY_DEF(int64_t, 3)
    ARRAY_DEF(uint8_t, 3)
    ARRAY_DEF(uint16_t, 3)
    ARRAY_DEF(uint32_t, 3)
    ARRAY_DEF(uint64_t, 3)
    ARRAY_DEF(float, 3)
    ARRAY_DEF(double, 3)
    ARRAY_DEF(std::complex<float>, 3)
    ARRAY_DEF(std::complex<double>, 3)
    ARRAY_DEF(bool, 4)
    ARRAY_DEF(int8_t, 4)
    ARRAY_DEF(int16_t, 4)
    ARRAY_DEF(int32_t, 4)
    ARRAY_DEF(int64_t, 4)
    ARRAY_DEF(uint8_t, 4)
    ARRAY_DEF(uint16_t, 4)
    ARRAY_DEF(uint32_t, 4)
    ARRAY_DEF(uint64_t, 4)
    ARRAY_DEF(float, 4)
    ARRAY_DEF(double, 4)
    ARRAY_DEF(std::complex<float>, 4)
    ARRAY_DEF(std::complex<double>, 4)
    ;
  
}
