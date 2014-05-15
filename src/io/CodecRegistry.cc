/**
 * @date Tue Oct 25 23:25:46 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Implements the CodecRegistry class.
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#include <vector>

#include <boost/filesystem.hpp>
#include <boost/format.hpp>

#include <bob/core/logging.h>
#include <bob/io/CodecRegistry.h>

boost::shared_ptr<bob::io::CodecRegistry> bob::io::CodecRegistry::instance() {
  static boost::shared_ptr<bob::io::CodecRegistry> s_instance(new CodecRegistry());
  return s_instance;
}

void bob::io::CodecRegistry::deregisterExtension(const char* ext) {
  s_extension2codec.erase(ext);
  s_extension2description.erase(ext);
}

const char* bob::io::CodecRegistry::getDescription(const char* ext) {
  auto it = s_extension2description.find(ext);
  if (it == s_extension2description.end()) return 0;
  return it->second.c_str();
}

void bob::io::CodecRegistry::deregisterFactory(bob::io::file_factory_t factory) {

  std::vector<std::string> to_remove;
  for (auto it = s_extension2codec.begin(); it != s_extension2codec.end(); ++it) {
    if (it->second == factory) to_remove.push_back(it->first);
  }

  for (auto it = to_remove.begin(); it != to_remove.end(); ++it) {
    s_extension2codec.erase(*it);
    s_extension2description.erase(*it);
  }

}

void bob::io::CodecRegistry::registerExtension(const char* extension,
    const char* description, bob::io::file_factory_t codec) {

  auto it = s_extension2codec.find(extension);

  if (it == s_extension2codec.end()) {
    s_extension2codec[extension] = codec;
    s_extension2description[extension] = description;
  }
  else if (!s_ignore) {
    boost::format m("extension already registered: %s - ignoring second registration with description `%s'");
    m % extension % description;
    bob::core::error << m.str() << std::endl;
    throw std::runtime_error(m.str());
  }

}

bool bob::io::CodecRegistry::isRegistered(const char* ext) {
  std::string extension(ext);
  std::string lower_extension = extension;
  std::transform(extension.begin(), extension.end(), lower_extension.begin(), ::tolower);
  return (s_extension2codec.find(lower_extension) != s_extension2codec.end());
}

bob::io::file_factory_t bob::io::CodecRegistry::findByExtension (const char* ext) {

  std::string extension(ext);
  std::string lower_extension = extension;
  std::transform(extension.begin(), extension.end(), lower_extension.begin(), ::tolower);

  std::map<std::string, bob::io::file_factory_t >::iterator it =
    s_extension2codec.find(lower_extension);

  if (it == s_extension2codec.end()) {
    boost::format m("unregistered extension: %s");
    m % lower_extension;
    throw std::runtime_error(m.str());
  }

  return it->second;

}

bob::io::file_factory_t bob::io::CodecRegistry::findByFilenameExtension
(const char* filename) {

  return findByExtension(boost::filesystem::path(filename).extension().c_str());

}
