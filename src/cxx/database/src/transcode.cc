/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Implements automatic transcoding functionality.
 */

#include "database/transcode.h"
#include "database/ArrayCodecRegistry.h"
#include "database/ArraysetCodecRegistry.h"

namespace db = Torch::database;

void db::array_transcode (const std::string& from, const std::string& to) {
  boost::shared_ptr<const ArrayCodec> fr_codec = db::ArrayCodecRegistry::getCodecByExtension(from);
  boost::shared_ptr<const ArrayCodec> to_codec = db::ArrayCodecRegistry::getCodecByExtension(to);
  to_codec->save(to, fr_codec->load(from));
}

void db::array_transcode (const std::string& from,
    const std::string& from_codecname, const std::string& to,
    const std::string& to_codecname) {
  boost::shared_ptr<const ArrayCodec> fr_codec = db::ArrayCodecRegistry::getCodecByName(from_codecname);
  boost::shared_ptr<const ArrayCodec> to_codec = db::ArrayCodecRegistry::getCodecByName(to_codecname);
  to_codec->save(to, fr_codec->load(from));
}

void db::arrayset_transcode (const std::string& from, const std::string& to) {
  boost::shared_ptr<const ArraysetCodec> fr_codec = db::ArraysetCodecRegistry::getCodecByExtension(from);
  boost::shared_ptr<const ArraysetCodec> to_codec = db::ArraysetCodecRegistry::getCodecByExtension(to);
  to_codec->save(to, fr_codec->load(from));
}

void db::arrayset_transcode (const std::string& from, 
    const std::string& from_codecname, const std::string& to, 
    const std::string& to_codecname) {
  boost::shared_ptr<const ArraysetCodec> fr_codec = db::ArraysetCodecRegistry::getCodecByName(from_codecname);
  boost::shared_ptr<const ArraysetCodec> to_codec = db::ArraysetCodecRegistry::getCodecByName(to_codecname);
  to_codec->save(to, fr_codec->load(from));
}
