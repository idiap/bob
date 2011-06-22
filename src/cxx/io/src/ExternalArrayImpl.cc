/**
 * @file io/src/ExternalArrayImpl.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Implements reading and writing data to arrays in files.
 */

#include <boost/filesystem.hpp>

#include "io/ExternalArrayImpl.h"
#include "io/InlinedArrayImpl.h"
#include "io/ArrayCodec.h"
#include "io/ArrayCodecRegistry.h"

namespace iod = Torch::io::detail;
namespace fs = boost::filesystem;

iod::ExternalArrayImpl::ExternalArrayImpl(const std::string& filename, 
    const std::string& codecname, bool newfile)
  : m_filename(fs::complete(filename).string())
{
  //the next instructions will raise an exception if the code is not found.
  if (codecname.size()) {
    m_codec = Torch::io::ArrayCodecRegistry::getCodecByName(codecname);
  }
  else {
    m_codec = Torch::io::ArrayCodecRegistry::getCodecByExtension(filename);
  }
  if (!newfile) reloadSpecification();
}

iod::ExternalArrayImpl::~ExternalArrayImpl() {}

void iod::ExternalArrayImpl::reloadSpecification () {
  m_codec->peek(m_filename, m_elementtype, m_ndim, m_shape);
}

void iod::ExternalArrayImpl::move(const std::string& filename,
    const std::string& codecname) {
  fs::path destination = fs::complete(filename);
  boost::shared_ptr<const Torch::io::ArrayCodec> newcodec;
  if (codecname.size()) 
    newcodec = Torch::io::ArrayCodecRegistry::getCodecByName(codecname);
  else 
    newcodec = Torch::io::ArrayCodecRegistry::getCodecByExtension(filename);
  if (newcodec == m_codec) { //just rename the file
    fs::rename(m_filename, destination);
  }
  else { //the user wants to re-write it in a different format. DON'T erase!
    newcodec->save(destination.string(), m_codec->load(m_filename));
    m_codec = newcodec;
  }
  m_filename = destination.string();
  reloadSpecification();
}

void iod::ExternalArrayImpl::set(const iod::InlinedArrayImpl& data) {
  m_codec->save(m_filename, data);
  reloadSpecification();
}
