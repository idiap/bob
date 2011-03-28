/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Implements an image format reader/writer using ffmpeg.
 *
 * This codec will only be able to work with 4D input and output (color videos)
 */

#include "database/VideoArrayCodec.h"
#include "database/ArrayCodecRegistry.h"
#include "database/Video.h"

namespace db = Torch::database;

//Takes care of the codec registration.
static bool register_codec() {
  db::ArrayCodecRegistry::addCodec(boost::shared_ptr<db::ArrayCodec>(new db::VideoArrayCodec())); 
  return true;
}

static bool codec_registered = register_codec();

db::VideoArrayCodec::VideoArrayCodec()
  : m_name("torch.video"),
    m_extensions()
{ 
  // subset extracted by executing "ffmpeg -formats"
  m_extensions.push_back(".avi");
  m_extensions.push_back(".dv");
  m_extensions.push_back(".filmstrip");
  m_extensions.push_back(".flv");
  m_extensions.push_back(".h261");
  m_extensions.push_back(".h263");
  m_extensions.push_back(".h264");
  m_extensions.push_back(".mov");
  m_extensions.push_back(".image2");
  m_extensions.push_back(".image2pipe");
  m_extensions.push_back(".m4v");
  m_extensions.push_back(".mjpeg");
  m_extensions.push_back(".mpeg");
  m_extensions.push_back(".mpegts");
  m_extensions.push_back(".ogg");
  m_extensions.push_back(".rawvideo");
  m_extensions.push_back(".rm");
  m_extensions.push_back(".rtsp");
  m_extensions.push_back(".yuv4mpegpipe");
}

db::VideoArrayCodec::~VideoArrayCodec() { }

void db::VideoArrayCodec::peek(const std::string& filename, 
    Torch::core::array::ElementType& eltype, size_t& ndim,
    size_t* shape) const 
{
  db::VideoReader v(filename);
  eltype = Torch::core::array::t_uint8;
  ndim = 4;
  shape[0] = v.numberOfFrames();
  shape[1] = 3;
  shape[2] = v.height();
  shape[3] = v.width();
}

db::detail::InlinedArrayImpl 
db::VideoArrayCodec::load(const std::string& filename) const {
  db::VideoReader v(filename);
  blitz::Array<uint8_t,4> retval;
  v.load(retval);
  return retval;
}

void db::VideoArrayCodec::save (const std::string& filename,
    const db::detail::InlinedArrayImpl& data) const {
  if (data.getNDim() != 4) throw db::DimensionError(data.getNDim(), 4);
  const blitz::Array<uint8_t,4>& array = data.get<uint8_t,4>();
  db::VideoWriter v(filename, array.extent(2), array.extent(3));
  v.append(array);
}
