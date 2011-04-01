/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Sat  5 Mar 20:26:56 2011 
 *
 * @brief Implementation of the Configuration main class
 */

#include <boost/filesystem.hpp>
#include "config/Configuration.h"
#include "config/PythonConfig.h"
#include "config/HDF5Config.h"

namespace conf = Torch::config;
namespace bp = boost::python;

conf::Configuration::Configuration(const std::string& s):
  m_dict()
{
  boost::filesystem::path p(s);
  //based on the extension, choose how to populate internal dictionary
  if (p.extension() == ".py") conf::detail::pyload(p, m_dict);
  else if (p.extension() == ".h5") conf::detail::hdf5load(p, m_dict);
  else throw conf::NotImplemented("load", p.extension());
}

conf::Configuration::Configuration(): 
  m_dict()
{
}

conf::Configuration::Configuration(const conf::Configuration& other):
  m_dict(other.m_dict)
{
}

conf::Configuration::~Configuration ()
{
}

conf::Configuration& conf::Configuration::operator= (const conf::Configuration& other) {
  m_dict = bp::dict();
  m_dict.update(other.m_dict);
  return *this;
}

void conf::Configuration::save (const std::string& s) const {
  boost::filesystem::path p(s);
  //based on the extension, choose how to populate internal dictionary
  if (p.extension() == ".py") conf::detail::pysave(p, m_dict);
  else if (p.extension() == ".h5") conf::detail::hdf5save(p, m_dict);
  else throw conf::NotImplemented("save", p.extension());
}

conf::Configuration& conf::Configuration::update (const conf::Configuration& other) {
  m_dict.update(other.m_dict);
  return *this;
}
