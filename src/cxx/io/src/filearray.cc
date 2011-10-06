/**
 * @file io/src/filearray.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Implements reading and writing data to arrays in files.
 */

#include <boost/filesystem.hpp>

#include "io/filearray.h"
#include "io/ArrayCodecRegistry.h"

#include "io/carray.h"

namespace io = Torch::io;
namespace fs = boost::filesystem;

io::fileinfo::fileinfo(const std::string& file, const std::string& codecname):
  filename(file)
{
  //the next instructions will raise an exception if the codec is not found.
  if (codecname.size()) {
    codec = Torch::io::ArrayCodecRegistry::getCodecByName(codecname);
  }
  else {
    codec = Torch::io::ArrayCodecRegistry::getCodecByExtension(filename);
  }

}

io::fileinfo::fileinfo(const fileinfo& other):
  filename(other.filename),
  codec(other.codec)
{
}

void io::fileinfo::read_type(typeinfo& info) {
  codec->peek(filename, info);
}

io::filearray::filearray(const std::string& filename, 
    const std::string& codecname, bool newfile):
  m_info(filename, codecname)
{
  if (!newfile) m_info.read_type(m_type);
}

io::filearray::~filearray() {}

void io::filearray::move(const std::string& filename,
    const std::string& codecname) {

  fs::path destination = fs::complete(filename);
  boost::shared_ptr<const Torch::io::ArrayCodec> newcodec;
  
  if (codecname.size())
    newcodec = Torch::io::ArrayCodecRegistry::getCodecByName(codecname);
  else 
    newcodec = Torch::io::ArrayCodecRegistry::getCodecByExtension(filename);

  if (newcodec == m_info.codec) { //just rename the file
    fs::rename(m_info.filename, destination);
  }
  
  else { //the user wants to re-write it in a different format. DON'T erase!
    io::carray tmp(m_type);
    m_info.codec->load(m_info.filename, tmp);
    newcodec->save(destination.string(), tmp);
    m_info.codec = newcodec;
  }

  m_info.filename = destination.string();
  m_info.read_type(m_type);
}

void io::filearray::load(io::buffer& array) const {
  return m_info.codec->load(m_info.filename, array);
}

void io::filearray::save(const io::buffer& data) {
  m_info.codec->save(m_info.filename, data);
  m_info.read_type(m_type);
}
