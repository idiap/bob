/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Implements an image format reader/writer using ffmpeg.
 *
 * This codec will only be able to work with 4D input and output (color videos)
 */

#include "io/VideoArrayCodec.h"
#include "io/ArrayCodecRegistry.h"
#include "io/Video.h"

namespace io = Torch::io;
namespace core = Torch::core;

//Takes care of the codec registration.
static bool register_codec() {
  io::ArrayCodecRegistry::addCodec(boost::shared_ptr<io::ArrayCodec>(new io::VideoArrayCodec())); 
  return true;
}

static bool codec_registered = register_codec();

io::VideoArrayCodec::VideoArrayCodec()
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

io::VideoArrayCodec::~VideoArrayCodec() { }

void io::VideoArrayCodec::peek(const std::string& filename, 
    io::typeinfo& info) const
{
  io::VideoReader v(filename);
  info = v.type();
}

void io::VideoArrayCodec::load(const std::string& file,
    io::buffer& array) const {
  io::VideoReader v(file);
  v.load(array);
}

void io::VideoArrayCodec::save (const std::string& file,
    const io::buffer& array) const {
  const io::typeinfo& type = array.type();
  if (type.nd != 4) throw io::DimensionError(type.nd, 4);
  io::VideoWriter v(file, type.shape[2], type.shape[3]);
  v.append(array);
}
