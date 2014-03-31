/**
 * @file io/python/matfile.cc
 * @date Tue Jan 28 15:56:02 CET 2014
 * @author Manuel Guenther <manuel.guenther@idiap.ch>
 *
 * @brief Bindings for reading matlab files.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#include <boost/python.hpp>
#include <bob/io/MatUtils.h>
#include <bob/python/ndarray.h>

using namespace boost::python;

static object read_matrix(const std::string& filename, const std::string& varname){
  // get type of data
  bob::core::array::typeinfo info;
  bob::io::detail::mat_peek(filename, info, varname);
  bob::python::py_array retval(info);

  // open matlab file
  boost::shared_ptr<mat_t> matfile(bob::io::detail::make_matfile(filename, MAT_ACC_RDONLY));
  if (!matfile){
    throw std::runtime_error("Could not find the given varname in the file");
  }
  bob::io::detail::read_array(matfile, retval, varname);

  return retval;
}

static object read_varnames(const std::string& filename){
  // get variable information
  auto lst = bob::io::detail::list_variables(filename);
  list varnames;
  for (auto it = lst->begin(); it != lst->end(); ++it){
    varnames.append(it->second.first);
  }

  return varnames;

}

void bind_matfile(){
  def("read_matlab_varnames", &read_varnames, arg("filename"), "Returns the list of varnames stored in the given matlab file.");
  def("read_matlab_matrix", &read_matrix, (arg("filename"), arg("varname")=""), "Reads the matlab matrix with the given varname from the given file. If a varname is not specified, the first matrix will be returned.");
}
