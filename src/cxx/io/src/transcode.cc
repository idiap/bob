/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Implements automatic transcoding functionality.
 */

#include "io/transcode.h"
#include "io/ArrayCodecRegistry.h"
#include "io/ArraysetCodecRegistry.h"

namespace io = Torch::io;

void io::array_transcode (const std::string& from, const std::string& to) {
  boost::shared_ptr<const ArrayCodec> fr_codec = io::ArrayCodecRegistry::getCodecByExtension(from);
  boost::shared_ptr<const ArrayCodec> to_codec = io::ArrayCodecRegistry::getCodecByExtension(to);
  to_codec->save(to, fr_codec->load(from));
}

void io::array_transcode (const std::string& from,
    const std::string& from_codecname, const std::string& to,
    const std::string& to_codecname) {
  boost::shared_ptr<const ArrayCodec> fr_codec = io::ArrayCodecRegistry::getCodecByName(from_codecname);
  boost::shared_ptr<const ArrayCodec> to_codec = io::ArrayCodecRegistry::getCodecByName(to_codecname);
  to_codec->save(to, fr_codec->load(from));
}

void io::arrayset_transcode (const std::string& from, const std::string& to) {
  boost::shared_ptr<const ArraysetCodec> fr_codec = io::ArraysetCodecRegistry::getCodecByExtension(from);
  boost::shared_ptr<const ArraysetCodec> to_codec = io::ArraysetCodecRegistry::getCodecByExtension(to);
  to_codec->save(to, fr_codec->load(from));
}

void io::arrayset_transcode (const std::string& from, 
    const std::string& from_codecname, const std::string& to, 
    const std::string& to_codecname) {
  boost::shared_ptr<const ArraysetCodec> fr_codec = io::ArraysetCodecRegistry::getCodecByName(from_codecname);
  boost::shared_ptr<const ArraysetCodec> to_codec = io::ArraysetCodecRegistry::getCodecByName(to_codecname);
  to_codec->save(to, fr_codec->load(from));
}
