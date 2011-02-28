/**
 * @file database/src/ExternalArrayImpl.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Implements reading and writing data to arrays in files.
 */

#include <boost/filesystem.hpp>

#include "database/ExternalArrayImpl.h"
#include "database/InlinedArrayImpl.h"
#include "database/ArrayCodec.h"
#include "database/ArrayCodecRegistry.h"

namespace tdd = Torch::database::detail;
namespace fs = boost::filesystem;

tdd::ExternalArrayImpl::ExternalArrayImpl(const std::string& filename, 
    const std::string& codecname, bool newfile)
  : m_filename(fs::complete(filename).string())
{
  //the next instructions will raise an exception if the code is not found.
  if (codecname.size()) {
    m_codec = Torch::database::ArrayCodecRegistry::getCodecByName(codecname);
  }
  else {
    m_codec = Torch::database::ArrayCodecRegistry::getCodecByExtension(filename);
  }
  if (!newfile) reloadSpecification();
}

tdd::ExternalArrayImpl::~ExternalArrayImpl() {}

void tdd::ExternalArrayImpl::reloadSpecification () {
  m_codec->peek(m_filename, m_elementtype, m_ndim, m_shape);
}

void tdd::ExternalArrayImpl::move(const std::string& filename,
    const std::string& codecname) {
  fs::path destination = fs::complete(filename);
  boost::shared_ptr<const Torch::database::ArrayCodec> newcodec;
  if (codecname.size()) 
    newcodec = Torch::database::ArrayCodecRegistry::getCodecByName(codecname);
  else 
    newcodec = Torch::database::ArrayCodecRegistry::getCodecByExtension(filename);
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

void tdd::ExternalArrayImpl::set(const tdd::InlinedArrayImpl& data) {
  m_codec->save(m_filename, data);
  reloadSpecification();
}
