/**
 * @file database/src/ArrayCodecRegistry.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Implements the ArrayCodecRegistry class.
 */

#include <boost/filesystem.hpp>

#include "database/ArrayCodecRegistry.h"
#include "database/Exception.h"

#include<iostream>

namespace db = Torch::database;

boost::shared_ptr<db::ArrayCodecRegistry> db::ArrayCodecRegistry::instance() {
  static boost::shared_ptr<db::ArrayCodecRegistry> s_instance(new ArrayCodecRegistry());
  return s_instance; 
}
    
void db::ArrayCodecRegistry::addCodec(boost::shared_ptr<db::ArrayCodec> codec) {
  boost::shared_ptr<ArrayCodecRegistry> instance = db::ArrayCodecRegistry::instance();
  std::map<std::string, boost::shared_ptr<db::ArrayCodec> >::iterator it = instance->s_name2codec.find(codec->name());

  if (it == instance->s_name2codec.end()) {
    instance->s_name2codec[codec->name()] = codec;
  }
  else {
    throw db::NameError(codec->name());
  }

  for (std::vector<std::string>::const_iterator jt = codec->extensions().begin(); jt != codec->extensions().end(); ++jt) {
    it = instance->s_extension2codec.find(*jt);
    if (it == instance->s_extension2codec.end()) {
      instance->s_extension2codec[*jt] = codec;
    }
    else {
      throw db::NameError(*jt);
    }
  }
}

boost::shared_ptr<const db::ArrayCodec>
db::ArrayCodecRegistry::getCodecByName(const std::string& name) {
  boost::shared_ptr<ArrayCodecRegistry> instance = db::ArrayCodecRegistry::instance();
  std::map<std::string, boost::shared_ptr<db::ArrayCodec> >::iterator it = instance->s_name2codec.find(name);
  if (it == instance->s_name2codec.end()) {
    throw db::CodecNotFound(name);
  }
  return it->second;
}

boost::shared_ptr<const db::ArrayCodec>
db::ArrayCodecRegistry::getCodecByExtension(const std::string& filename)
{
  boost::shared_ptr<ArrayCodecRegistry> instance = db::ArrayCodecRegistry::instance();
  boost::filesystem::path path(filename);
  std::map<std::string, boost::shared_ptr<db::ArrayCodec> >::iterator it = instance->s_extension2codec.find(path.extension());
  if (it == instance->s_extension2codec.end()) {
    throw db::ExtensionNotRegistered(path.extension());
  }
  return it->second;
}
