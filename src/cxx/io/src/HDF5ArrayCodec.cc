/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 * @date Sun 17 Apr 11:24:32 2011 CEST
 *
 * @brief Implements the HDF5 (.hdf5) array codec
 */

#include <boost/shared_array.hpp>
#include <boost/filesystem.hpp>

#include "io/HDF5ArrayCodec.h"
#include "io/HDF5File.h"
#include "io/ArrayCodecRegistry.h"
#include "io/HDF5Exception.h"

namespace io = Torch::io;

//Takes care of the codec registration.
static bool register_codec() {
  io::ArrayCodecRegistry::addCodec(boost::shared_ptr<io::ArrayCodec>(new io::HDF5ArrayCodec())); 
  return true;
}

static bool codec_registered = register_codec();
static boost::shared_ptr<io::HDF5Error> init = io::HDF5Error::instance();

/**
 * Chooses the best format to read from.
 *
 * If the size on the descriptor is bigger than 1, the user is reading either
 * an arrayset or an HDF5 produced outside torch. In this particular case, read
 * everything in one shot.
 */
static const io::HDF5Descriptor& 
choose_format(const std::vector<io::HDF5Descriptor>& fmt) {
  size_t def = 0; ///< by default, we use the first format

  if (!fmt.at(def).expandable || fmt[def].size > 1) {
    //saved as array or created outside torch
    def += 1;
  }

  return fmt.at(def);
}

io::HDF5ArrayCodec::HDF5ArrayCodec()
  : m_name("hdf5.array.binary"),
    m_extensions()
{ 
  m_extensions.push_back(".h5");
  m_extensions.push_back(".hdf5");
}

io::HDF5ArrayCodec::~HDF5ArrayCodec() { }

void io::HDF5ArrayCodec::peek(const std::string& filename, 
    io::typeinfo& info) const {

  io::HDF5File f(filename, io::HDF5File::in);
  std::vector<std::string> paths;
  f.paths(paths);
  if (!paths.size()) throw io::HDF5InvalidPath(filename, "/array");
  choose_format(f.describe(paths[0])).type.copy_to(info);
}

void io::HDF5ArrayCodec::load(const std::string& file, buffer& array) const {

  io::HDF5File f(file, io::HDF5File::in);
  std::vector<std::string> paths;
  f.paths(paths);
  if (!paths.size()) throw io::HDF5InvalidPath(file, "/array");

  const std::string& name = paths[0];
  const io::HDF5Type& descr = choose_format(f.describe(name)).type;

  io::typeinfo info;
  descr.copy_to(info);
  if(!array.type().is_compatible(info)) array.set(info);

  f.read_buffer(name, 0, array);
}

void io::HDF5ArrayCodec::save (const std::string& filename,
    const io::buffer& data) const {

  static std::string varname("array");

  io::HDF5File f(filename, io::HDF5File::trunc);
  f.create(varname, data.type(), false, 0); ///straight array, no compression
  f.write_buffer(varname, 0, data); 
}
