/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed 26 Oct 08:58:42 2011 CEST
 *
 * @brief Implements the BinaryArrayFile type 
 */

#include <boost/make_shared.hpp>

#include "io/BinFile.h"
#include "io/CodecRegistry.h"

namespace io = Torch::io;

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

    virtual const io::typeinfo& array_type () const {
      return m_type;
    }

    virtual const io::typeinfo& arrayset_type () const {
      return m_type;
    }

    virtual size_t arrayset_size() const {
      return m_file.size();
    }

    virtual const std::string& name() const {
      return s_codecname;
    }

    virtual void array_read(io::buffer& buffer) {

      if(!m_file)
        throw std::runtime_error("uninitialized binary file cannot be read");

      m_file.read(0, buffer);

    }

    virtual void arrayset_read(io::buffer& buffer, size_t index) {

      if(!m_file)
        throw std::runtime_error("uninitialized binary file cannot be read");

      m_file.read(index, buffer);

    }

    virtual size_t arrayset_append (const io::buffer& buffer) {

      m_file.write(buffer);

      if (arrayset_size() == 1) m_type = buffer.type();

      return arrayset_size() - 1;

    }
    
    virtual void array_write (const io::buffer& buffer) {

      //we don't have a special way to treat write()'s like in HDF5.
      arrayset_append(buffer);

    }

  private: //representation

    io::BinFile m_file;
    io::typeinfo m_type;
    std::string m_filename;

    static std::string s_codecname;

};

std::string BinaryArrayFile::s_codecname = "torch.binary";

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
  
  instance->registerExtension(".bin", &make_file);

  return true;

}

static bool codec_registered = register_codec();
