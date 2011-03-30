/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Sat  5 Mar 20:26:56 2011 
 *
 * @brief Implementation of the Configuration main class
 */

#include <boost/filesystem.hpp>
#include "config/HDF5Config.h"
#include "config/Exception.h"

namespace conf = Torch::config;
namespace bp = boost::python;

void conf::detail::hdf5load(const boost::filesystem::path& path,
    bp::dict& dict) {
  throw conf::NotImplemented("save", path.extension());
}

void conf::detail::hdf5save(const boost::filesystem::path& path,
    const bp::dict& dict) {
  throw conf::NotImplemented("save", path.extension());
}
