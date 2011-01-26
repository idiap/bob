/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Wed 26 Jan 2011 07:46:17
 *
 * @brief Python bindings to Torch::core::BinInputFile and Torch::core::BinOutputFile
 */

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>

#include "core/BinInputFile.h"
#include "core/BinOutputFile.h"

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

static const char* ARRAY_READ_DOC = "Adapts the size of each dimension of the passed blitz array to the ones of the binary file and copies the data for a certain array in the file to the passed blitz::Array.";
#define ARRAY_READ_DEF(T,D) .def("bzread", (void (db::BinInputFile::*)(size_t, blitz::Array<T,D>&))&db::BinInputFile::read<T,D>, (arg("self"), arg("index"), arg("array")), ARRAY_READ_DOC)

static const char* ARRAY_WRITE_DOC = "Writes a single blitz::Array<> into the binary file. Please note that this array should conform to the shape and element type of the arrays already inserted. If no array was inserted, the element type and shape will be defined when you first write an array to this binary file.";
#define ARRAY_WRITE_DEF(T,D) .def("bzwrite", (void (db::BinOutputFile::*)(const blitz::Array<T,D>&))&db::BinOutputFile::write<T,D>, (arg("self"), arg("array")), ARRAY_WRITE_DOC)

void bind_database_binfile() {
  class_<db::BinInputFile, boost::shared_ptr<db::BinInputFile>, boost::noncopyable>("BinInputFile", "An BinInputFile allows users to read data from files containing standard Torch binary coded data", init<const std::string&>((arg("filename")), "Initializes an binary file reader. Please note that this constructor will not load the data."))
    .add_property("shape", &get_shape<db::BinInputFile>, "The shape of arrays in this binary file. Please note all arrays in the file have necessarily the same shape.")
    .add_property("__len__", &db::BinInputFile::getNSamples, "The number of arrays in this binary file.")
    ARRAY_READ_DEF(bool, 1)
    ARRAY_READ_DEF(int8_t, 1)
    ARRAY_READ_DEF(int16_t, 1)
    ARRAY_READ_DEF(int32_t, 1)
    ARRAY_READ_DEF(int64_t, 1)
    ARRAY_READ_DEF(uint8_t, 1)
    ARRAY_READ_DEF(uint16_t, 1)
    ARRAY_READ_DEF(uint32_t, 1)
    ARRAY_READ_DEF(uint64_t, 1)
    ARRAY_READ_DEF(float, 1)
    ARRAY_READ_DEF(double, 1)
    ARRAY_READ_DEF(std::complex<float>, 1)
    ARRAY_READ_DEF(std::complex<double>, 1)
    ARRAY_READ_DEF(bool, 2)
    ARRAY_READ_DEF(int8_t, 2)
    ARRAY_READ_DEF(int16_t, 2)
    ARRAY_READ_DEF(int32_t, 2)
    ARRAY_READ_DEF(int64_t, 2)
    ARRAY_READ_DEF(uint8_t, 2)
    ARRAY_READ_DEF(uint16_t, 2)
    ARRAY_READ_DEF(uint32_t, 2)
    ARRAY_READ_DEF(uint64_t, 2)
    ARRAY_READ_DEF(float, 2)
    ARRAY_READ_DEF(double, 2)
    ARRAY_READ_DEF(std::complex<float>, 2)
    ARRAY_READ_DEF(std::complex<double>, 2)
    ARRAY_READ_DEF(bool, 3)
    ARRAY_READ_DEF(int8_t, 3)
    ARRAY_READ_DEF(int16_t, 3)
    ARRAY_READ_DEF(int32_t, 3)
    ARRAY_READ_DEF(int64_t, 3)
    ARRAY_READ_DEF(uint8_t, 3)
    ARRAY_READ_DEF(uint16_t, 3)
    ARRAY_READ_DEF(uint32_t, 3)
    ARRAY_READ_DEF(uint64_t, 3)
    ARRAY_READ_DEF(float, 3)
    ARRAY_READ_DEF(double, 3)
    ARRAY_READ_DEF(std::complex<float>, 3)
    ARRAY_READ_DEF(std::complex<double>, 3)
    ARRAY_READ_DEF(bool, 4)
    ARRAY_READ_DEF(int8_t, 4)
    ARRAY_READ_DEF(int16_t, 4)
    ARRAY_READ_DEF(int32_t, 4)
    ARRAY_READ_DEF(int64_t, 4)
    ARRAY_READ_DEF(uint8_t, 4)
    ARRAY_READ_DEF(uint16_t, 4)
    ARRAY_READ_DEF(uint32_t, 4)
    ARRAY_READ_DEF(uint64_t, 4)
    ARRAY_READ_DEF(float, 4)
    ARRAY_READ_DEF(double, 4)
    ARRAY_READ_DEF(std::complex<float>, 4)
    ARRAY_READ_DEF(std::complex<double>, 4)
    ;
  
  class_<db::BinOutputFile, boost::shared_ptr<db::BinOutputFile>, boost::noncopyable>("BinOutputFile", "An BinOutputFile allows users to write data to files containing standard Torch binary coded data", init<const std::string&, optional<bool> >((arg("filename"), arg("append")=false), "Initializes an binary file writer. If the file exists, it is truncated unless the append attribute is set to True."))
    .add_property("shape", &get_shape<db::BinOutputFile>, "The shape of arrays in this binary file. Please note all arrays in the file have necessarily the same shape.")
    .add_property("__len__", &db::BinOutputFile::getNSamples, "The number of arrays in this binary file.")
    ARRAY_WRITE_DEF(bool, 1)
    ARRAY_WRITE_DEF(int8_t, 1)
    ARRAY_WRITE_DEF(int16_t, 1)
    ARRAY_WRITE_DEF(int32_t, 1)
    ARRAY_WRITE_DEF(int64_t, 1)
    ARRAY_WRITE_DEF(uint8_t, 1)
    ARRAY_WRITE_DEF(uint16_t, 1)
    ARRAY_WRITE_DEF(uint32_t, 1)
    ARRAY_WRITE_DEF(uint64_t, 1)
    ARRAY_WRITE_DEF(float, 1)
    ARRAY_WRITE_DEF(double, 1)
    ARRAY_WRITE_DEF(std::complex<float>, 1)
    ARRAY_WRITE_DEF(std::complex<double>, 1)
    ARRAY_WRITE_DEF(bool, 2)
    ARRAY_WRITE_DEF(int8_t, 2)
    ARRAY_WRITE_DEF(int16_t, 2)
    ARRAY_WRITE_DEF(int32_t, 2)
    ARRAY_WRITE_DEF(int64_t, 2)
    ARRAY_WRITE_DEF(uint8_t, 2)
    ARRAY_WRITE_DEF(uint16_t, 2)
    ARRAY_WRITE_DEF(uint32_t, 2)
    ARRAY_WRITE_DEF(uint64_t, 2)
    ARRAY_WRITE_DEF(float, 2)
    ARRAY_WRITE_DEF(double, 2)
    ARRAY_WRITE_DEF(std::complex<float>, 2)
    ARRAY_WRITE_DEF(std::complex<double>, 2)
    ARRAY_WRITE_DEF(bool, 3)
    ARRAY_WRITE_DEF(int8_t, 3)
    ARRAY_WRITE_DEF(int16_t, 3)
    ARRAY_WRITE_DEF(int32_t, 3)
    ARRAY_WRITE_DEF(int64_t, 3)
    ARRAY_WRITE_DEF(uint8_t, 3)
    ARRAY_WRITE_DEF(uint16_t, 3)
    ARRAY_WRITE_DEF(uint32_t, 3)
    ARRAY_WRITE_DEF(uint64_t, 3)
    ARRAY_WRITE_DEF(float, 3)
    ARRAY_WRITE_DEF(double, 3)
    ARRAY_WRITE_DEF(std::complex<float>, 3)
    ARRAY_WRITE_DEF(std::complex<double>, 3)
    ARRAY_WRITE_DEF(bool, 4)
    ARRAY_WRITE_DEF(int8_t, 4)
    ARRAY_WRITE_DEF(int16_t, 4)
    ARRAY_WRITE_DEF(int32_t, 4)
    ARRAY_WRITE_DEF(int64_t, 4)
    ARRAY_WRITE_DEF(uint8_t, 4)
    ARRAY_WRITE_DEF(uint16_t, 4)
    ARRAY_WRITE_DEF(uint32_t, 4)
    ARRAY_WRITE_DEF(uint64_t, 4)
    ARRAY_WRITE_DEF(float, 4)
    ARRAY_WRITE_DEF(double, 4)
    ARRAY_WRITE_DEF(std::complex<float>, 4)
    ARRAY_WRITE_DEF(std::complex<double>, 4)
    ;
}
