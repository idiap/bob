/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Mon  7 Feb 21:34:22 2011 
 *
 * @brief Implements the BinaryArrayCodec type 
 */

#include "database/BinaryArrayCodec.h"
#include "database/BinFile.h"

namespace db = Torch::database;

db::BinaryArrayCodec::BinaryArrayCodec()
  : m_name("torch.array.binary"),
    m_extensions()
{ 
  m_extensions.push_back(".bin");
}

db::BinaryArrayCodec::~BinaryArrayCodec() { }

void db::BinaryArrayCodec::peek(const std::string& filename, 
    Torch::core::array::ElementType& eltype, size_t& ndim) const {
  db::BinFile f(filename, db::BinFile::in);
  eltype = f.getElementType();
  ndim = f.getNDimensions();
}

db::detail::InlinedArrayImpl 
db::BinaryArrayCodec::load(const std::string& filename) const {
  db::BinFile f(filename, db::BinFile::in);
  eltype = f.getElementType();
  ndim = f.getNDimensions();
}

void db::BinaryArrayCodec::save (const std::string& filename, 
    const db::detail::InlinedArrayImpl& data) const {
}

const std::string& dbname () const;

/**
 * Returns a list of known extensions this codec can handle. The
 * extensions include the initial ".". So, to cover for jpeg images, you
 * may return a vector containing ".jpeg" and ".jpg" for example. Case
 * matters, so ".jpeg" and ".JPEG" are different extensions. If are the
 * responsible to cover all possible variations an extension can have.
 */
const std::vector<std::string>& extensions () const;
