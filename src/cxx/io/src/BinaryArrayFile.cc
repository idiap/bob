/**
 * @file cxx/io/src/BinaryArrayFile.cc
 * @date Wed Oct 26 17:11:16 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Implements the BinaryArrayFile type
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <boost/make_shared.hpp>

#include "io/BinFile.h"
#include "io/CodecRegistry.h"

namespace io = bob::io;
namespace ca = bob::core::array;

class BinaryArrayFile: public io::File {

  public: //api

    BinaryArrayFile(const std::string& path, io::BinFile::openmode mode):
      m_file(path, mode),
      m_filename(path) {
        if (m_file.size()) m_type.set(m_file.getElementType(), 
            m_file.getNDimensions(), m_file.getShape());
      }

    virtual ~BinaryArrayFile() { }

    virtual const std::string& filename() const {
      return m_filename;
    }

    virtual const ca::typeinfo& array_type () const {
      return m_type;
    }

    virtual const ca::typeinfo& arrayset_type () const {
      return m_type;
    }

    virtual size_t arrayset_size() const {
      return m_file.size();
    }

    virtual const std::string& name() const {
      return s_codecname;
    }

    virtual void array_read(ca::interface& buffer) {

      if(!m_file)
        throw std::runtime_error("uninitialized binary file cannot be read");

      m_file.read(0, buffer);

    }

    virtual void arrayset_read(ca::interface& buffer, size_t index) {

      if(!m_file)
        throw std::runtime_error("uninitialized binary file cannot be read");

      m_file.read(index, buffer);

    }

    virtual size_t arrayset_append (const ca::interface& buffer) {

      m_file.write(buffer);

      if (arrayset_size() == 1) m_type = buffer.type();

      return arrayset_size() - 1;

    }
    
    virtual void array_write (const ca::interface& buffer) {

      //we don't have a special way to treat write()'s like in HDF5.
      arrayset_append(buffer);

    }

  private: //representation

    io::BinFile m_file;
    ca::typeinfo m_type;
    std::string m_filename;

    static std::string s_codecname;

};

std::string BinaryArrayFile::s_codecname = "bob.binary";

/**
 * From this point onwards we have the registration procedure. If you are
 * looking at this file for a coding example, just follow the procedure bellow,
 * minus local modifications you may need to apply.
 */

/**
 * This defines the factory method F that can create codecs of this type.
 * 
 * Here are the meanings of the mode flag that should be respected by your
 * factory implementation:
 *
 * 'r': opens for reading only - no modifications can occur; it is an
 *      error to open a file that does not exist for read-only operations.
 * 'w': opens for reading and writing, but truncates the file if it
 *      exists; it is not an error to open files that do not exist with
 *      this flag. 
 * 'a': opens for reading and writing - any type of modification can 
 *      occur. If the file does not exist, this flag is effectively like
 *      'w'.
 *
 * Returns a newly allocated File object that can read and write data to the
 * file using a specific backend.
 *
 * @note: This method can be static.
 */
static boost::shared_ptr<io::File> 
make_file (const std::string& path, char mode) {

  io::BinFile::openmode _mode;
  if (mode == 'r') _mode = io::BinFile::in;
  else if (mode == 'w') _mode = io::BinFile::out;
  else if (mode == 'a') _mode = io::BinFile::append;
  else throw std::invalid_argument("unsupported binary (.bin) file opening mode");

  return boost::make_shared<BinaryArrayFile>(path, _mode);

}

/**
 * Takes care of codec registration per se.
 */
static bool register_codec() {

  boost::shared_ptr<io::CodecRegistry> instance =
    io::CodecRegistry::instance();
  
  instance->registerExtension(".bin", "bob alpha binary data format (DEPRECATED)", &make_file);

  return true;

}

static bool codec_registered = register_codec();
