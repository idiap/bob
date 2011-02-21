/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Sun 20 Feb 09:55:32 2011 
 *
 * @brief Implements the matlab (.mat) arrayset codec using matio
 */

#include "database/MatArraysetCodec.h"
#include "database/ArraysetCodecRegistry.h"
#include "database/Exception.h"

namespace db = Torch::database;

//Takes care of the codec registration.
static bool register_codec() {
  db::ArraysetCodecRegistry::addCodec(boost::shared_ptr<db::ArraysetCodec>(new db::MatArraysetCodec())); 
  return true;
}

static bool codec_registered = register_codec(); 

db::MatArraysetCodec::MatArraysetCodec()
  : m_name("matlab.arrayset.binary"),
    m_extensions()
{ 
  m_extensions.push_back(".mat");
}

db::MatArraysetCodec::~MatArraysetCodec() { }

void db::MatArraysetCodec::peek(const std::string& filename, 
    Torch::core::array::ElementType& eltype, size_t& ndim,
    size_t* shape, size_t& samples) const {
}

db::detail::InlinedArraysetImpl db::MatArraysetCodec::load
(const std::string& filename) const {
}

db::Array db::MatArraysetCodec::load
(const std::string& filename, size_t id) const {
}

void db::MatArraysetCodec::append
(const std::string& filename, const Array& array) const {
}

void db::MatArraysetCodec::save (const std::string& filename, 
    const detail::InlinedArraysetImpl& data) const {
}
