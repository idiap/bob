/**
 * @file src/core/python/src/tensorfile.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Binds the TensorFile object type into python 
 */

#include <boost/python.hpp>

#include "core/Tensor.h"
#include "core/TensorFile.h"

using namespace boost::python;

static int tfh_get_size(const Torch::TensorFile::Header& tfh, int i) {
  if (i < tfh.m_n_dimensions) return tfh.m_size[i];
  return 0;
}

static void tfh_set_size(Torch::TensorFile::Header& tfh, int i, int value) {
  if (i < tfh.m_n_dimensions) tfh.m_size[i] = value;
}

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(openwrite_overloads, openWrite, 4, 7)

//opens a new file for writing and set the header properties to be like the
//tensor given as model
static bool tf_open_write(Torch::TensorFile& f, const char* name, 
    const Torch::Tensor& model) {
  switch (model.nDimension()) {
    case 1:
      return f.openWrite(name, model.getDatatype(), model.nDimension(), 
          model.size(0)); 
      break;
    case 2:
      return f.openWrite(name, model.getDatatype(), model.nDimension(), 
          model.size(0), model.size(1)); 
      break;
    case 3:
      return f.openWrite(name, model.getDatatype(), model.nDimension(), 
          model.size(0), model.size(1), model.size(2)); 
      break;
    case 4:
      return f.openWrite(name, model.getDatatype(), model.nDimension(), 
          model.size(0), model.size(1), model.size(2), model.size(3)); 
      break;
    default:
      return false;
  }
  return false;
}

void bind_core_tensorfile()
{
  class_<Torch::TensorFile::Header>("TensorFileHeader", "Header information in TensorFiles", no_init)
    .def("getTensorIndex", &Torch::TensorFile::Header::getTensorIndex, (arg("self"), arg("tensor_index")), "Get the index of some tensor in the file")
    .def("update", &Torch::TensorFile::Header::update, (arg("self")), "Updates internal information")
    .def_readwrite("type", &Torch::TensorFile::Header::m_type)
    .def_readwrite("n_samples", &Torch::TensorFile::Header::m_n_samples)
    .def_readwrite("n_dimensions", &Torch::TensorFile::Header::m_n_dimensions)
    .def_readwrite("tensorSize", &Torch::TensorFile::Header::m_tensorSize)
    .add_property("size", &tfh_get_size, &tfh_set_size)
    ;

  class_<Torch::TensorFile>("TensorFile", "TensorFiles help users in loading and saving tensor data into files", init<>())
    .def("openRead", &Torch::TensorFile::openRead, (arg("self"), arg("filename")), "Opens an existing tensor file to read")
    .def("openWrite", &tf_open_write, (arg("self"), arg("filename"), arg("tensor_model")), "Opens a new tensor file for writing. If the file exists, truncates it. The new tensor file will be set to have the dimensions of the model tensor passed as the second parameter.")
    .def("openWrite", &Torch::TensorFile::openWrite, openwrite_overloads((arg("filename"), arg("tensor_type"), arg("number_of_dimensions"), arg("size_dim0"), arg("size_dim1")=0, arg("size_dim2")=0, arg("size_dim3")=0), "Opens a new tensor file for writing. If the file exists, truncates it."))
    .def("openAppend", &Torch::TensorFile::openAppend, (arg("self"), arg("filename")), "Opens an existing file to append")
    .def("close", &Torch::TensorFile::close, (arg("self")), "Closes an opened file")
    .def("isOpened", &Torch::TensorFile::isOpened, (arg("self")), "Tells if the current file is opened")
    .def("load", (Torch::Tensor* (Torch::TensorFile::*)(void))&Torch::TensorFile::load, return_value_policy<manage_new_object>(), arg("self"), "Loads the next available tensor in the file and returns a new object.")
    .def("load", (Torch::Tensor* (Torch::TensorFile::*)(int index))&Torch::TensorFile::load, return_value_policy<manage_new_object>(), (arg("self"), arg("index")), "Loads the i-th available tensor in the file and returns a new object.")
    .def("load", (bool (Torch::TensorFile::*)(Torch::Tensor&))&Torch::TensorFile::load, (arg("self"), arg("tensor")), "Loads the next available tensor in the file into the given tensor object. This method will be more efficient than allocating a new tensor every time as no re-allocations will occur if the given tensor specifications are right.")
    .def("load", (bool (Torch::TensorFile::*)(Torch::Tensor&, int))&Torch::TensorFile::load, (arg("self"), arg("tensor"), arg("index")), "Loads the i-th available tensor in the file into the given tensor object. This method will be more efficient than allocating a new tensor every time as no re-allocations will occur if the given tensor specifications are right.")
    .def("save", &Torch::TensorFile::save, (arg("self"), arg("tensor")), "Saves the given tensor in the file")
    .def("getHeader", &Torch::TensorFile::getHeader, return_internal_reference<>(), (arg("self")), "Gives user acccess to the current file header")
    ;
}
