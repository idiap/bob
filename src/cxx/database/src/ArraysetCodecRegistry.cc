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

std::map<std::string, boost::shared_ptr<db::ArraysetCodec> > db::ArraysetCodecRegistry::s_name2codec;
std::map<std::string, boost::shared_ptr<db::ArraysetCodec> > db::ArraysetCodecRegistry::s_extension2codec;
    
void db::ArraysetCodecRegistry::addCodec
(boost::shared_ptr<db::ArraysetCodec> codec) {
  std::map<std::string, boost::shared_ptr<db::ArraysetCodec> >::iterator it = s_name2codec.find(codec->name());
  if (it == s_name2codec.end()) {
    s_name2codec[codec->name()] = codec;
  }
  else {
    throw db::IndexError();
  }

  for (std::vector<std::string>::const_iterator jt = codec->extensions().begin(); jt != codec->extensions().end(); ++jt) {
    it = s_extension2codec.find(*jt);
    if (it == s_extension2codec.end()) {
      s_extension2codec[*jt] = codec;
    }
    else {
      throw db::IndexError();
    }
  }
}

boost::shared_ptr<const db::ArraysetCodec>
db::ArraysetCodecRegistry::getCodecByName(const std::string& name) {
  std::map<std::string, boost::shared_ptr<db::ArraysetCodec> >::iterator it = s_name2codec.find(name);
  if (it == s_name2codec.end()) {
    throw db::IndexError();
  }
  return it->second;
}

boost::shared_ptr<const db::ArraysetCodec>
db::ArraysetCodecRegistry::getCodecByExtension(const std::string& filename)
{
  boost::filesystem::path path(filename);
  std::map<std::string, boost::shared_ptr<db::ArraysetCodec> >::iterator it = s_extension2codec.find(path.extension());
  if (it == s_extension2codec.end()) {
    throw db::IndexError();
  }
  return it->second;
}
