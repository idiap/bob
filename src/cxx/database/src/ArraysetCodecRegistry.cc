/**
 * @file database/src/ArraysetCodecRegistry.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Implements the ArraysetCodecRegistry class.
 */

#include <boost/filesystem.hpp>

#include "database/ArraysetCodecRegistry.h"
#include "database/dataset_common.h"

namespace db = Torch::database;

boost::shared_ptr<db::ArraysetCodecRegistry> db::ArraysetCodecRegistry::instance() {
  static boost::shared_ptr<db::ArraysetCodecRegistry> s_instance(new ArraysetCodecRegistry());
  return s_instance; 
}    
    
void db::ArraysetCodecRegistry::addCodec(boost::shared_ptr<db::ArraysetCodec> codec) {
  boost::shared_ptr<ArraysetCodecRegistry> instance = db::ArraysetCodecRegistry::instance();
  std::map<std::string, boost::shared_ptr<db::ArraysetCodec> >::iterator it = instance->s_name2codec.find(codec->name());
  if (it == instance->s_name2codec.end()) {
    instance->s_name2codec[codec->name()] = codec;
  }
  else {
    throw db::IndexError();
  }

  for (std::vector<std::string>::const_iterator jt = codec->extensions().begin(); jt != codec->extensions().end(); ++jt) {
    it = instance->s_extension2codec.find(*jt);
    if (it == instance->s_extension2codec.end()) {
      instance->s_extension2codec[*jt] = codec;
    }
    else {
      throw db::IndexError();
    }
  }
}

boost::shared_ptr<const db::ArraysetCodec>
db::ArraysetCodecRegistry::getCodecByName(const std::string& name) {
  boost::shared_ptr<ArraysetCodecRegistry> instance = db::ArraysetCodecRegistry::instance();
  std::map<std::string, boost::shared_ptr<db::ArraysetCodec> >::iterator it = instance->s_name2codec.find(name);
  if (it == instance->s_name2codec.end()) {
    throw db::IndexError();
  }
  return it->second;
}

boost::shared_ptr<const db::ArraysetCodec>
db::ArraysetCodecRegistry::getCodecByExtension(const std::string& filename)
{
  boost::shared_ptr<ArraysetCodecRegistry> instance = db::ArraysetCodecRegistry::instance();
  boost::filesystem::path path(filename);
  std::map<std::string, boost::shared_ptr<db::ArraysetCodec> >::iterator it = instance->s_extension2codec.find(path.extension());
  if (it == instance->s_extension2codec.end()) {
    throw db::IndexError();
  }
  return it->second;
}
