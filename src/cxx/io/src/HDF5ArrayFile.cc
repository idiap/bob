/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 * @date Sun 17 Apr 11:24:32 2011 CEST
 *
 * @brief Implements the HDF5 (.hdf5) array codec
 */

#include <boost/make_shared.hpp>
#include <boost/filesystem.hpp>

#include "io/CodecRegistry.h"

#include "io/HDF5File.h"
#include "io/HDF5Exception.h"

namespace fs = boost::filesystem;
namespace io = Torch::io;
namespace ca = Torch::core::array;

/**
 * Required initialization by the HDF5 sub-system
 */
static boost::shared_ptr<io::HDF5Error> init = io::HDF5Error::instance();

/**
 * Read and write arrays in HDF5 format
 */
class HDF5ArrayFile: public io::File {

  public:

    HDF5ArrayFile (const std::string& filename, io::HDF5File::mode_t mode):
      m_file(filename, mode), 
      m_filename(filename),
      m_size_arrayset(0),
      m_newfile(true) { 

        //tries to update the current descriptors
        std::vector<std::string> paths;
        m_file.paths(paths);
        
        if (paths.size()) { //file contains data, read it and establish defaults
          m_path = paths[0]; ///< locks on a path name from now on...
          m_newfile = false; ///< blocks re-initialization

          //arrayset reading
          const io::HDF5Descriptor& desc_arrayset = m_file.describe(m_path)[0];
          desc_arrayset.type.copy_to(m_type_arrayset);
          m_size_arrayset = desc_arrayset.size;

          //array reading
          m_file.describe(m_path)[1].type.copy_to(m_type_array);

          //if m_type_all has extent == 1 on the first dimension, collapse that
          if (m_type_array.shape[0] == 1) m_type_array = m_type_arrayset;
        }

        else {
          //default path in case the file is new or has been truncated
          m_path = "/array";
        }

      }

    virtual ~HDF5ArrayFile() { }

    virtual const std::string& filename() const {
      return m_filename;
    }

    virtual const ca::typeinfo& array_type () const {
      return m_type_array;
    }

    virtual const ca::typeinfo& arrayset_type () const {
      return m_type_arrayset;
    }

    virtual size_t arrayset_size() const {
      return m_size_arrayset;
    }

    virtual const std::string& name() const {
      return s_codecname;
    }

    virtual void array_read(ca::interface& buffer) {

      if(m_newfile) 
        throw std::runtime_error("uninitialized HDF5 file cannot be read");

      if(!buffer.type().is_compatible(m_type_array)) buffer.set(m_type_array);

      m_file.read_buffer(m_path, 0, buffer);
    }

    virtual void arrayset_read(ca::interface& buffer, size_t index) {

      if(m_newfile) 
        throw std::runtime_error("uninitialized HDF5 file cannot be read");

      if(!buffer.type().is_compatible(m_type_arrayset)) buffer.set(m_type_arrayset);

      m_file.read_buffer(m_path, index, buffer);
    }

    virtual size_t arrayset_append (const ca::interface& buffer) {

      if (m_newfile) {
        //creates non-compressible, extensible dataset on HDF5 file
        m_newfile = false;
        m_file.create(m_path, buffer.type(), true, 0);
        m_file.describe(m_path)[0].type.copy_to(m_type_arrayset);
        m_file.describe(m_path)[1].type.copy_to(m_type_array);

        //if m_type_all has extent == 1 on the first dimension, collapse that
        if (m_type_array.shape[0] == 1) m_type_array = m_type_arrayset;
      }

      m_file.extend_buffer(m_path, buffer);
      ++m_size_arrayset;
      //needs to flush the data to the file
      return m_size_arrayset - 1; ///< index of this object in the file

    }

    virtual void array_write (const ca::interface& buffer) {

      if (!m_newfile) {
        throw std::runtime_error("cannot perform single (array-style) write on file/dataset that have already been initialized -- try to use a new file");
      }

      m_newfile = false;
      m_file.create(m_path, buffer.type(), false, 0);

      m_file.describe(m_path)[0].type.copy_to(m_type_arrayset);
      m_file.describe(m_path)[1].type.copy_to(m_type_array);

      //if m_type_all has extent == 1 on the first dimension, collapse that
      if (m_type_array.shape[0] == 1) m_type_array = m_type_arrayset;

      //otherwise, all must be in place...
      m_file.write_buffer(m_path, 0, buffer);
    }

  private: //representation
    
    io::HDF5File m_file;
    std::string  m_filename;
    ca::typeinfo m_type_array;    ///< type for reading all data at once
    ca::typeinfo m_type_arrayset; ///< type for reading data by sub-arrays
    size_t       m_size_arrayset; ///< number of arrays in arrayset mode
    std::string  m_path; ///< default path to use
    bool         m_newfile; ///< path check optimization

    static std::string  s_codecname;

};

std::string HDF5ArrayFile::s_codecname = "torch.hdf5";

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

  io::HDF5File::mode_t h5mode;
  if (mode == 'r') h5mode = io::HDF5File::in;
  else if (mode == 'w') h5mode = io::HDF5File::trunc;
  else if (mode == 'a') h5mode = io::HDF5File::inout;
  else throw std::invalid_argument("unsupported file opening mode");

  return boost::make_shared<HDF5ArrayFile>(path, h5mode);

}

/**
 * Takes care of codec registration per se.
 */
static bool register_codec() {

  boost::shared_ptr<io::CodecRegistry> instance =
    io::CodecRegistry::instance();
  
  instance->registerExtension(".h5", &make_file);
  instance->registerExtension(".hdf5", &make_file);
  instance->registerExtension(".hdf", &make_file);

  return true;

}

static bool codec_registered = register_codec();
