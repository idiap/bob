/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Sun 20 Feb 09:51:30 2011 
 *
 * @brief Implements the matlab (.mat) array codec using matio
 */

#include <boost/shared_array.hpp>
#include <boost/filesystem.hpp>

#include "io/MatArrayCodec.h"
#include "io/MatUtils.h"
#include "io/ArrayCodecRegistry.h"
#include "io/Exception.h"

namespace io = Torch::io;

//Takes care of the codec registration.
static bool register_codec() {
  io::ArrayCodecRegistry::addCodec(boost::shared_ptr<io::ArrayCodec>(new io::MatArrayCodec())); 
  return true;
}

static bool codec_registered = register_codec();

io::MatArrayCodec::MatArrayCodec()
  : m_name("matlab.array.binary"),
    m_extensions()
{ 
  m_extensions.push_back(".mat");
}

io::MatArrayCodec::~MatArrayCodec() { }

void io::MatArrayCodec::peek(const std::string& filename, 
    io::typeinfo& info) const {
  io::detail::mat_peek(filename, info);
  if (info.nd == 0 || info.nd > 4) 
    throw io::DimensionError(info.nd, TORCH_MAX_DIM);
  if (info.dtype == Torch::core::array::t_unknown) 
    throw io::UnsupportedTypeError(info.dtype);
}

void io::MatArrayCodec::load(const std::string& filename,
    io::buffer& array) const {
  boost::shared_ptr<mat_t> mat = io::detail::make_matfile(filename,
      MAT_ACC_RDONLY);
  if (!mat) throw io::FileNotReadable(filename);
  io::detail::read_array(mat, array);
}

void io::MatArrayCodec::save (const std::string& filename, 
    const io::buffer& array) const {

  static std::string varname("array");

  //this file is supposed to hold a single array. delete it if it exists
  boost::filesystem::path path (filename);
  if (boost::filesystem::exists(path)) boost::filesystem::remove(path);

  boost::shared_ptr<mat_t> mat = io::detail::make_matfile(filename, 
      MAT_ACC_RDWR);
  if (!mat) throw io::FileNotReadable(filename);

  io::detail::write_array(mat, varname, array);

}
