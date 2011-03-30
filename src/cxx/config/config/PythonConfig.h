/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Defines the API for the Python Configuration functionality
 */

#ifndef TORCH_CONFIG_PYTHONCONFIG_H
#define TORCH_CONFIG_PYTHONCONFIG_H

#include <boost/python.hpp>
#include <boost/filesystem.hpp>

namespace Torch { namespace config { namespace detail {

  /**
   * This method loads into a boost::python dictionary, the configuration
   * written in a python file.
   */
  void pyload(const boost::filesystem::path& path, boost::python::dict& dict);

  /**
   * This method saves into a "pickled" python file, the configuration
   * estabilished in a dictionary. Note that all objects captured by the
   * set<>() calls to the Configuration object should be pickle'able in this
   * case.
   */
  void pysave(const boost::filesystem::path& path, 
      const boost::python::dict& dict);

}}}

#endif /* TORCH_CONFIG_PYTHONCONFIG_H */
