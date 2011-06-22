/**
 * @file io/src/ArrayCodecRegistry.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Implements the ArrayCodecRegistry class.
 */

#include <boost/filesystem.hpp>

#include "io/ArrayCodecRegistry.h"
#include "io/Exception.h"

#include<iostream>

namespace io = Torch::io;

boost::shared_ptr<io::ArrayCodecRegistry> io::ArrayCodecRegistry::instance() {
  static boost::shared_ptr<io::ArrayCodecRegistry> s_instance(new ArrayCodecRegistry());
  return s_instance; 
}
    
void io::ArrayCodecRegistry::removeCodecByName(const std::string& codecname) {
  boost::shared_ptr<ArrayCodecRegistry> instance = io::ArrayCodecRegistry::instance();
  std::map<std::string, boost::shared_ptr<io::ArrayCodec> >::iterator it = instance->s_name2codec.find(codecname);

  if (it == instance->s_name2codec.end()) {
    throw io::NameError(codecname);
  }
  //remove all extensions
  for (std::vector<std::string>::const_iterator jt = it->second->extensions().begin(); jt != it->second->extensions().end(); ++jt) {
    instance->s_extension2codec.erase(*jt);
  }
  //remove the codec itself
  instance->s_name2codec.erase(it->first);
}

void io::ArrayCodecRegistry::addCodec(boost::shared_ptr<io::ArrayCodec> codec) {
  boost::shared_ptr<ArrayCodecRegistry> instance = io::ArrayCodecRegistry::instance();
  std::map<std::string, boost::shared_ptr<io::ArrayCodec> >::iterator it = instance->s_name2codec.find(codec->name());

  if (it == instance->s_name2codec.end()) {
    instance->s_name2codec[codec->name()] = codec;
  }
  else {
    throw io::NameError(codec->name());
  }

  for (std::vector<std::string>::const_iterator jt = codec->extensions().begin(); jt != codec->extensions().end(); ++jt) {
    it = instance->s_extension2codec.find(*jt);
    if (it == instance->s_extension2codec.end()) {
      instance->s_extension2codec[*jt] = codec;
    }
    else {
      throw io::NameError(*jt);
    }
  }
}

boost::shared_ptr<const io::ArrayCodec>
io::ArrayCodecRegistry::getCodecByName(const std::string& name) {
  boost::shared_ptr<ArrayCodecRegistry> instance = io::ArrayCodecRegistry::instance();
  std::map<std::string, boost::shared_ptr<io::ArrayCodec> >::iterator it = instance->s_name2codec.find(name);
  if (it == instance->s_name2codec.end()) {
    throw io::CodecNotFound(name);
  }
  return it->second;
}

boost::shared_ptr<const io::ArrayCodec>
io::ArrayCodecRegistry::getCodecByExtension(const std::string& filename)
{
  boost::shared_ptr<ArrayCodecRegistry> instance = io::ArrayCodecRegistry::instance();
  boost::filesystem::path path(filename);
  std::map<std::string, boost::shared_ptr<io::ArrayCodec> >::iterator it = instance->s_extension2codec.find(path.extension());
  if (it == instance->s_extension2codec.end()) {
    throw io::ExtensionNotRegistered(path.extension());
  }
  return it->second;
}
