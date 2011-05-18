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
  m_current_path("/"), m_dict()
{
  boost::filesystem::path p(s);
  //based on the extension, choose how to populate internal dictionary
  if (p.extension() == ".py") conf::detail::pyload(p, m_dict);
  else if (p.extension() == ".hdf5") conf::detail::hdf5load(p, m_dict);
  else throw conf::NotImplemented("load", p.extension());
}

conf::Configuration::Configuration(): 
  m_current_path("/"), m_dict()
{
}

conf::Configuration::Configuration(const conf::Configuration& other):
  m_current_path("/"), m_dict(other.m_dict)
{
}

conf::Configuration::~Configuration ()
{
}

conf::Configuration& conf::Configuration::operator= (const conf::Configuration& other) {
  m_dict = bp::dict();
  m_dict.update(other.m_dict);
  m_current_path = other.m_current_path;
  return *this;
}

void conf::Configuration::save (const std::string& s) const {
  boost::filesystem::path p(s);
  //based on the extension, choose how to populate internal dictionary
  if (p.extension() == ".py") conf::detail::pysave(p, m_dict);
  else if (p.extension() == ".hdf5") conf::detail::hdf5save(p, m_dict);
  else throw conf::NotImplemented("save", p.extension());
}

// Function adapted from
// http://stackoverflow.com/questions/1746136/how-do-i-normalize-a-pathname-using-boostfilesystem/1750710#1750710
static boost::filesystem::path normalize(const boost::filesystem::path& p) {
  boost::filesystem::path result;
  for(boost::filesystem::path::iterator it=p.begin(); it!=p.end(); ++it) {
    if(*it == "..") {
      result = result.parent_path();
    }
    else if(*it == ".") {
      // Ignore
    }
    else {
      // Just cat other path entries
      result /= *it;
    }
  }
  
  return result;
}

conf::Configuration& conf::Configuration::update (const conf::Configuration& other) {
  m_dict.update(other.m_dict);
  return *this;
}

void conf::Configuration::cd(std::string path) {
  if (!path.empty() && path.at(0) == '/') {
    // Path is absolute
    m_current_path = normalize(boost::filesystem::path(path));
  }
  else {
    // Path is relative
    m_current_path = normalize(m_current_path / path);
  }
}

std::string conf::Configuration::getAbsolutePath(const std::string path) const {
  if (!path.empty() && path.at(0) == '/') {
    // Path is absolute
    return normalize(boost::filesystem::path(path)).string().substr(1);
  }
  else {
    // Path is relative
    return normalize(m_current_path / path).string().substr(1);
  }
}
