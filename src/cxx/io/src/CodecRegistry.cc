/**
 * @file io/src/CodecRegistry.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Implements the CodecRegistry class.
 */

#include <vector>

#include <boost/filesystem.hpp>
#include <boost/format.hpp>

#include "io/CodecRegistry.h"

#include<iostream>

namespace io = Torch::io;

boost::shared_ptr<io::CodecRegistry> io::CodecRegistry::instance() {
  static boost::shared_ptr<io::CodecRegistry> s_instance(new CodecRegistry());
  return s_instance; 
}
    
void io::CodecRegistry::deregisterExtension(const std::string& ext) {
  s_extension2codec.erase(ext);
  s_extension2description.erase(ext);
}

void io::CodecRegistry::deregisterFactory(io::file_factory_t factory) {

  std::vector<std::string> to_remove;
  for (std::map<std::string, io::file_factory_t>::iterator
      it = s_extension2codec.begin(); it != s_extension2codec.end(); ++it) {
    if (it->second == factory) to_remove.push_back(it->first);
  }

  for (std::vector<std::string>::const_iterator it = to_remove.begin(); 
      it != to_remove.end(); ++it) {
    s_extension2codec.erase(*it);
    s_extension2description.erase(*it);
  }

}

void io::CodecRegistry::registerExtension(const std::string& extension,
    const std::string& description, io::file_factory_t codec) {

  std::map<std::string, io::file_factory_t>::iterator it = 
    s_extension2codec.find(extension);

  if (it == s_extension2codec.end()) {
    s_extension2codec[extension] = codec;
    s_extension2description[extension] = description;
  }
  else {
    boost::format m("extension already registered: %s");
    m % extension;
    throw std::runtime_error(m.str().c_str());
  }

}

io::file_factory_t io::CodecRegistry::findByExtension
(const std::string& extension) {

  std::map<std::string, io::file_factory_t >::iterator it = 
    s_extension2codec.find(extension);

  if (it == s_extension2codec.end()) {
    boost::format m("unregistered extension: %s");
    m % extension;
    throw std::runtime_error(m.str().c_str());
  }

  return it->second;

}

io::file_factory_t io::CodecRegistry::findByFilenameExtension
(const std::string& filename) {

  boost::filesystem::path path(filename);
  return findByExtension(path.extension());

}

boost::shared_ptr<io::File> io::open (const std::string& filename, 
  const std::string& pretend_extension, char mode) {

  boost::shared_ptr<io::CodecRegistry> instance =
    io::CodecRegistry::instance();

  if (pretend_extension.size()) 
    return instance->findByExtension(pretend_extension)(filename, mode);

  else
    return instance->findByFilenameExtension(filename)(filename, mode);

}
