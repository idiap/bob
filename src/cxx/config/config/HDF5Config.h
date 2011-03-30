/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Defines the API for the HDF5-based Configuration functionality
 */

#ifndef TORCH_CONFIG_HDF5CONFIG_H
#define TORCH_CONFIG_HDF5CONFIG_H

#include <boost/python.hpp>
#include <boost/filesystem.hpp>

namespace Torch { namespace config { namespace detail {

  /**
   * This method loads into a boost::python dictionary, the configuration
   * written in a HDF5 file.
   */
  void hdf5load(const boost::filesystem::path& path, boost::python::dict& dict);

  /**
   * This method saves into a HDF5 file, the configuration estabilished in a
   * dictionary. Note that all objects captured by the set<>() calls to the
   * Configuration object should have python bindings, so that we can apply
   * boost::python::extract<T>() on them.
   */
  void hdf5save(const boost::filesystem::path& path, const boost::python::dict&
      dict);

}}}

#endif /* TORCH_CONFIG_HDF5CONFIG_H */
