/**
 * @file database/src/ExternalArraysetImpl.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Implements reading and writing data to arrays in files.
 */

#include <boost/filesystem.hpp>

#include "database/ExternalArraysetImpl.h"
#include "database/InlinedArraysetImpl.h"
#include "database/ArraysetCodecRegistry.h"

namespace tdd = Torch::database::detail;

tdd::ExternalArraysetImpl::ExternalArraysetImpl(const std::string& filename, 
    const std::string& codecname)
  : m_filename(filename)
{
  //the next instructions will raise an exception if the code is not found.
  if (codecname.size()) {
    m_codec = Torch::database::ArraysetCodecRegistry::getCodecByName(codecname);
  }
  else {
    m_codec = Torch::database::ArraysetCodecRegistry::getCodecByExtension(filename);
  }
}

tdd::ExternalArraysetImpl::~ExternalArraysetImpl() {}

void tdd::ExternalArraysetImpl::getSpecification
(Torch::core::array::ElementType& eltype, size_t& ndim, size_t* shape, size_t& samples) const {
  m_codec->peek(m_filename, eltype, ndim, shape, samples);
}

void tdd::ExternalArraysetImpl::move(const std::string& filename,
    const std::string& codecname) {
  boost::shared_ptr<const Torch::database::ArraysetCodec> newcodec;
  if (codecname.size())
    newcodec = Torch::database::ArraysetCodecRegistry::getCodecByName(codecname);
  else
    newcodec = Torch::database::ArraysetCodecRegistry::getCodecByExtension(filename);
  if (newcodec == m_codec) { //just rename the file
    boost::filesystem::path path(m_filename);
    boost::filesystem::rename(boost::filesystem::path(m_filename),
        boost::filesystem::path(filename));
    m_filename = filename;
  }
  else { //the user wants to re-write it in a different format.
    newcodec->save(filename, m_codec->load(m_filename));
    boost::filesystem::remove(boost::filesystem::path(m_filename));
    m_codec = newcodec;
  }
}

Torch::database::Array tdd::ExternalArraysetImpl::operator[] (size_t id) const {
  return m_codec->load(m_filename, id);
}

void tdd::ExternalArraysetImpl::add
(boost::shared_ptr<const Torch::database::Array> array) {
  add(*array.get()); 
}

void tdd::ExternalArraysetImpl::add(const Torch::database::Array& array) {
  m_codec->append(m_filename, array);
}

void tdd::ExternalArraysetImpl::add(const tdd::InlinedArraysetImpl& set) {
  for(std::list<boost::shared_ptr<Torch::database::Array> >::const_iterator it= set.arrays().begin(); it != set.arrays().end(); ++it) {
    add(*it);
  }
}

tdd::InlinedArraysetImpl tdd::ExternalArraysetImpl::load() const {
  return m_codec->load(m_filename);
}

void tdd::ExternalArraysetImpl::save(const InlinedArraysetImpl& set) {
  m_codec->save(m_filename, set);
}
