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
namespace fs = boost::filesystem;

tdd::ExternalArraysetImpl::ExternalArraysetImpl(const std::string& filename, 
    const std::string& codecname, bool newfile)
  : m_filename(fs::complete(filename).string())
{
  //the next instructions will raise an exception if the code is not found.
  if (codecname.size()) {
    m_codec = Torch::database::ArraysetCodecRegistry::getCodecByName(codecname);
  }
  else {
    m_codec = Torch::database::ArraysetCodecRegistry::getCodecByExtension(filename);
  }
  if (!newfile) reloadSpecification();
}

tdd::ExternalArraysetImpl::~ExternalArraysetImpl() {}

void tdd::ExternalArraysetImpl::reloadSpecification() {
  m_codec->peek(m_filename, m_elementtype, m_ndim, m_shape, m_samples);
}

void tdd::ExternalArraysetImpl::move(const std::string& filename,
    const std::string& codecname) {
  fs::path destination = fs::complete(filename);
  boost::shared_ptr<const Torch::database::ArraysetCodec> newcodec;
  if (codecname.size())
    newcodec = Torch::database::ArraysetCodecRegistry::getCodecByName(codecname);
  else
    newcodec = Torch::database::ArraysetCodecRegistry::getCodecByExtension(filename);
  if (newcodec == m_codec) { //just rename the file
    fs::rename(m_filename, destination);
  }
  else { //the user wants to re-write it in a different format. DON'T erase!
    newcodec->save(filename, m_codec->load(m_filename));
    m_codec = newcodec;
  }
  m_filename = destination.string();
  reloadSpecification();
}

Torch::database::Array tdd::ExternalArraysetImpl::operator[] (size_t id) const {
  return m_codec->load(m_filename, id);
}

void tdd::ExternalArraysetImpl::checkCompatibility(const Torch::database::Array& array) const {
  if (m_elementtype != Torch::core::array::t_unknown) {
    if (array.getElementType() != m_elementtype) throw Torch::database::TypeError(array.getElementType(), m_elementtype);
    if (array.getNDim() != m_ndim) throw Torch::database::DimensionError(array.getNDim(), m_ndim);
    for (size_t i=0; i<m_ndim; ++i)
      if (array.getShape()[i] != m_shape[i]) throw Torch::database::DimensionError(array.getShape()[i], m_shape[i]);
  }
}

size_t tdd::ExternalArraysetImpl::add
(boost::shared_ptr<const Torch::database::Array> array) {
  return add(*array.get()); 
}

size_t tdd::ExternalArraysetImpl::add(const Torch::database::Array& array) {
  checkCompatibility(array);
  m_codec->append(m_filename, array);
  reloadSpecification();
  return m_samples;
}

void tdd::ExternalArraysetImpl::extend(const tdd::InlinedArraysetImpl& set) {
  for(std::map<size_t, boost::shared_ptr<Torch::database::Array> >::const_iterator it= set.index().begin(); it != set.index().end(); ++it) {
    add(it->second);
  }
  reloadSpecification();
}

void tdd::ExternalArraysetImpl::remove(size_t id) {
  if (id > m_samples) throw Torch::database::IndexError(id);
  //loads the file and rewrite it.
  //TODO: Optimize to avoid loading the whole file in memory
  tdd::InlinedArraysetImpl data = get();
  data.remove(id);
  set(data);
  reloadSpecification();
}

void tdd::ExternalArraysetImpl::add(size_t id, 
    boost::shared_ptr<const Torch::database::Array> array) {
  add(id, *array.get());
}

void tdd::ExternalArraysetImpl::add(size_t id,
    const Torch::database::Array& array) {
  if (id != (m_samples+1)) throw Torch::database::IndexError(id);
  add(array);
  reloadSpecification();
}

void tdd::ExternalArraysetImpl::set(size_t id, 
    boost::shared_ptr<const Torch::database::Array> array) {
  set(id, *array.get());
}

void tdd::ExternalArraysetImpl::set(size_t id,
    const Torch::database::Array& array) {
  if (id > m_samples) throw Torch::database::IndexError(id);
  //loads the file and rewrite it.
  //TODO: Optimize to avoid loading the whole file in memory
  tdd::InlinedArraysetImpl data = get();
  data[id] = array;
  set(data);
  reloadSpecification();
}

tdd::InlinedArraysetImpl tdd::ExternalArraysetImpl::get() const {
  return m_codec->load(m_filename);
}

void tdd::ExternalArraysetImpl::set(const InlinedArraysetImpl& set) {
  m_codec->save(m_filename, set);
  reloadSpecification();
}

bool tdd::ExternalArraysetImpl::exists(size_t id) const {
  return (id <= getNSamples());
}
