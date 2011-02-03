/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 * @date Wed 26 Jan 2011 07:46:17
 *
 * @brief Python bindings to Torch::core::BinFile
 */

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/preprocessor/stringize.hpp>

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


static const char* ARRAY_READ_DOC = "Reads data in the binary file and return a blitz::Array with a copy of this data.";
static const char* ARRAY_WRITE_DOC = "Writes a single blitz::Array<> into the binary file. Please note that this array should conform to the shape and element type of the arrays already inserted. If no array was inserted, the element type and shape will be defined when you first write an array to this binary file.";
#define ARRAY_DEF(T,N,D) .def(BOOST_PP_STRINGIZE(read_ ## N ## _ ## D), (blitz::Array<T,D> (db::BinFile::*)(size_t))&db::BinFile::read<T,D>, (arg("self"), arg("index")), ARRAY_READ_DOC) \
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
    ARRAY_DEF(bool, bool, 1)
    ARRAY_DEF(int8_t, int8, 1)
    ARRAY_DEF(int16_t, int16, 1)
    ARRAY_DEF(int32_t, int32, 1)
    ARRAY_DEF(int64_t, int64, 1)
    ARRAY_DEF(uint8_t, uint8, 1)
    ARRAY_DEF(uint16_t, uint16, 1)
    ARRAY_DEF(uint32_t, uint32, 1)
    ARRAY_DEF(uint64_t, uint64, 1)
    ARRAY_DEF(float, float32, 1)
    ARRAY_DEF(double, float64, 1)
    ARRAY_DEF(std::complex<float>, complex64, 1)
    ARRAY_DEF(std::complex<double>, complex128, 1)
    ARRAY_DEF(bool, bool, 2)
    ARRAY_DEF(int8_t, int8, 2)
    ARRAY_DEF(int16_t, int16, 2)
    ARRAY_DEF(int32_t, int32, 2)
    ARRAY_DEF(int64_t, int64, 2)
    ARRAY_DEF(uint8_t, uint8, 2)
    ARRAY_DEF(uint16_t, uint16, 2)
    ARRAY_DEF(uint32_t, uint32, 2)
    ARRAY_DEF(uint64_t, uint64, 2)
    ARRAY_DEF(float, float32, 2)
    ARRAY_DEF(double, float64, 2)
    ARRAY_DEF(std::complex<float>, complex64, 2)
    ARRAY_DEF(std::complex<double>, complex128, 2)
    ARRAY_DEF(bool, bool, 3)
    ARRAY_DEF(int8_t, int8, 3)
    ARRAY_DEF(int16_t, int16, 3)
    ARRAY_DEF(int32_t, int32, 3)
    ARRAY_DEF(int64_t, int64, 3)
    ARRAY_DEF(uint8_t, uint8, 3)
    ARRAY_DEF(uint16_t, uint16, 3)
    ARRAY_DEF(uint32_t, uint32, 3)
    ARRAY_DEF(uint64_t, uint64, 3)
    ARRAY_DEF(float, float32, 3)
    ARRAY_DEF(double, float64, 3)
    ARRAY_DEF(std::complex<float>, complex64, 3)
    ARRAY_DEF(std::complex<double>, complex128, 3)
    ARRAY_DEF(bool, bool, 4)
    ARRAY_DEF(int8_t, int8, 4)
    ARRAY_DEF(int16_t, int16, 4)
    ARRAY_DEF(int32_t, int32, 4)
    ARRAY_DEF(int64_t, int64, 4)
    ARRAY_DEF(uint8_t, uint8, 4)
    ARRAY_DEF(uint16_t, uint16, 4)
    ARRAY_DEF(uint32_t, uint32, 4)
    ARRAY_DEF(uint64_t, uint64, 4)
    ARRAY_DEF(float, float32, 4)
    ARRAY_DEF(double, float64, 4)
    ARRAY_DEF(std::complex<float>, complex64, 4)
    ARRAY_DEF(std::complex<double>, complex128, 4)
    ;

}
