/**
 * @file io/src/ExternalArraysetImpl.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Implements reading and writing data to arrays in files.
 */

#include <boost/filesystem.hpp>

#include "io/ExternalArraysetImpl.h"
#include "io/InlinedArraysetImpl.h"
#include "io/ArraysetCodecRegistry.h"

namespace iod = Torch::io::detail;
namespace fs = boost::filesystem;

iod::ExternalArraysetImpl::ExternalArraysetImpl(const std::string& filename, 
    const std::string& codecname, bool newfile)
  : m_filename(fs::complete(filename).string())
{
  //the next instructions will raise an exception if the code is not found.
  if (codecname.size()) {
    m_codec = Torch::io::ArraysetCodecRegistry::getCodecByName(codecname);
  }
  else {
    m_codec = Torch::io::ArraysetCodecRegistry::getCodecByExtension(filename);
  }
  if (!newfile) reloadSpecification();
}

iod::ExternalArraysetImpl::~ExternalArraysetImpl() {}

void iod::ExternalArraysetImpl::reloadSpecification() {
  m_codec->peek(m_filename, m_elementtype, m_ndim, m_shape, m_samples);
}

void iod::ExternalArraysetImpl::move(const std::string& filename,
    const std::string& codecname) {
  fs::path destination = fs::complete(filename);
  boost::shared_ptr<const Torch::io::ArraysetCodec> newcodec;
  if (codecname.size())
    newcodec = Torch::io::ArraysetCodecRegistry::getCodecByName(codecname);
  else
    newcodec = Torch::io::ArraysetCodecRegistry::getCodecByExtension(filename);
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

Torch::io::Array iod::ExternalArraysetImpl::operator[] (size_t id) const {
  return m_codec->load(m_filename, id);
}

void iod::ExternalArraysetImpl::checkCompatibility(const Torch::io::Array& array) const {
  if (m_elementtype != Torch::core::array::t_unknown) {
    if (array.getElementType() != m_elementtype) throw Torch::io::TypeError(array.getElementType(), m_elementtype);
    if (array.getNDim() != m_ndim) throw Torch::io::DimensionError(array.getNDim(), m_ndim);
    for (size_t i=0; i<m_ndim; ++i)
      if (array.getShape()[i] != m_shape[i]) throw Torch::io::DimensionError(array.getShape()[i], m_shape[i]);
  }
}

size_t iod::ExternalArraysetImpl::add
(boost::shared_ptr<const Torch::io::Array> array) {
  return add(*array.get()); 
}

size_t iod::ExternalArraysetImpl::add(const Torch::io::Array& array) {
  checkCompatibility(array);
  m_codec->append(m_filename, array);
  reloadSpecification();
  return m_samples;
}

void iod::ExternalArraysetImpl::extend(const iod::InlinedArraysetImpl& set) {
  for(size_t i=0; i<set.size(); ++i) add(set[i]);
  reloadSpecification();
}

void iod::ExternalArraysetImpl::remove(size_t id) {
  if (id > m_samples) throw Torch::io::IndexError(id);
  //loads the file and rewrite it.
  //TODO: Optimize to avoid loading the whole file in memory
  iod::InlinedArraysetImpl data = get();
  data.remove(id);
  set(data);
  reloadSpecification();
}

void iod::ExternalArraysetImpl::set(size_t id, 
    boost::shared_ptr<const Torch::io::Array> array) {
  set(id, *array.get());
}

void iod::ExternalArraysetImpl::set(size_t id,
    const Torch::io::Array& array) {
  if (id > m_samples) throw Torch::io::IndexError(id);
  //loads the file and rewrite it.
  //TODO: Optimize to avoid loading the whole file in memory
  iod::InlinedArraysetImpl data = get();
  data[id] = array;
  set(data);
  reloadSpecification();
}

iod::InlinedArraysetImpl iod::ExternalArraysetImpl::get() const {
  return m_codec->load(m_filename);
}

void iod::ExternalArraysetImpl::set(const InlinedArraysetImpl& set) {
  m_codec->save(m_filename, set);
  reloadSpecification();
}
