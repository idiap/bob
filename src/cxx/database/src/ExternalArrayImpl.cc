/**
 * @file database/src/ExternalArrayImpl.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Implements reading and writing data to arrays in files.
 */

#include <boost/filesystem.hpp>

#include "database/ExternalArrayImpl.h"
#include "database/InlinedArrayImpl.h"
#include "database/ArrayCodecRegistry.h"

namespace tdd = Torch::database::detail;

tdd::ExternalArrayImpl::ExternalArrayImpl(const std::string& filename, 
    const std::string& codecname)
  : m_filename(filename)
{
  //the next instructions will raise an exception if the code is not found.
  if (codecname.size()) {
    m_codec = Torch::database::ArrayCodecRegistry::getCodecByName(codecname);
  }
  else {
    m_codec = Torch::database::ArrayCodecRegistry::getCodecByExtension(filename);
  }
}

tdd::ExternalArrayImpl::~ExternalArrayImpl() {}

void tdd::ExternalArrayImpl::getSpecification
(Torch::core::array::ElementType& eltype, size_t& ndim,
 size_t& shape[Torch::core::array::N_MAX_DIMENSIONS_ARRAY]) const {
  m_codec->peek(m_filename, eltype, ndim, shape);
}

void tdd::ExternalArrayImpl::move(const std::string& filename,
    const std::string& codecname) {
  boost::shared_ptr<const Torch::database::ArrayCodec> newcodec;
  if (codecname.size()) 
    newcodec = Torch::database::ArrayCodecRegistry::getCodecByName(codecname);
  else 
    newcodec = Torch::database::ArrayCodecRegistry::getCodecByExtension(filename);
  if (newcodec == m_codec) { //just rename the file
    boost::filesystem::path path(m_filename);
    boost::filesystem::rename(boost::filesystem::path(m_filename),
        boost::filesystem::path(filename));
    m_filename = filename;
  }
  else { //the user wants to re-write it in a different format.
    newcodec->save(filename, m_codec->load(m_filename));
    boost::filesystem::remove(boost::filesystem::path(m_filename));
  }
}
