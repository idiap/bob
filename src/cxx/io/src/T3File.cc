/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed 26 Oct 11:45:00 2011 CEST
 *
 * @brief Implements a Torch3vision bindata reader/writer
 *
 * The format, as described in the old source code goes like this.
 *
 * 1) data is always recorded in little endian format
 * 2) the first 4 bytes describe an integer that indicates the number of arrays
 * to follow
 * 3) the second 4 bytes describe an integer that specifies the frame width.
 * 4) all arrays inserted there are single dimensional arrays.
 * 5) all elements from all arrays are "normally" float (4-bytes), but could be
 * double if set in the header of T3 during compilation. The file size will
 * indicate the right type to use.
 *
 * Because of this restriction, this codec will only be able to work with
 * single-dimension input.
 */

#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/make_shared.hpp>

//some infrastructure to check the file size
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "core/array_check.h"
#include "io/CodecRegistry.h"
#include "io/Exception.h"

namespace fs = boost::filesystem;
namespace io = Torch::io;
namespace ca = Torch::core::array;

static inline size_t get_filesize(const std::string& filename) {
  struct stat filestatus;
  stat(filename.c_str(), &filestatus);
  return filestatus.st_size;
}

class T3File: public io::File {

  public: //api

    T3File(const std::string& path, char mode):
      m_filename(path),
      m_newfile(true),
      m_length(0) {
        if ( mode == 'r' || (mode == 'a' && fs::exists(path) ) ) { // try peek
          size_t fsize = get_filesize(path);
          fsize -= 8; // remove the first two entries
          // read the first two 4-byte integers in the file, convert to unsigned
          
          std::fstream s(path.c_str(), std::ios::binary|std::ios::in);

          if (!s) throw io::FileNotReadable(path);
          
          uint32_t nsamples, framesize;
          nsamples = framesize = 0;
          s.read((char*)&nsamples, sizeof(uint32_t));
          s.read((char*)&framesize, sizeof(uint32_t));

          m_length = nsamples;
          
          // are those floats or doubles?
          if (fsize == (nsamples*framesize*sizeof(float))) {
            m_type_array.dtype = Torch::core::array::t_float32;
            m_type_arrayset.dtype = Torch::core::array::t_float32;
          }
          else if (fsize == (nsamples*framesize*sizeof(double))) {
            m_type_array.dtype = Torch::core::array::t_float64;
            m_type_arrayset.dtype = Torch::core::array::t_float64;
          }
          else 
            throw io::TypeError(Torch::core::array::t_float32, Torch::core::array::t_unknown);

          size_t shape[2] = {nsamples, framesize};
          m_type_array.set_shape<size_t>(2, &shape[0]);
          m_type_arrayset.set_shape<size_t>(1, &shape[1]);
          m_newfile = false;

        }
      }

    virtual ~T3File() { }

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
      return m_length;
    }

    virtual const std::string& name() const {
      return s_codecname;
    }

    virtual void array_read(ca::interface& buffer) {

      if (m_newfile) throw std::runtime_error("cannot read uninitialized t3 binary file");

      if (!buffer.type().is_compatible(m_type_array)) buffer.set(m_type_array);

      //open the file, now for reading the contents...
      std::ifstream ifile(m_filename.c_str(), std::ios::binary|std::ios::in);

      //skip the first 8 bytes, that contain the header that we already read
      ifile.seekg(8);
      ifile.read(static_cast<char*>(buffer.ptr()), buffer.type().buffer_size());

    }

    virtual void arrayset_read(ca::interface& buffer, size_t index) {

      if (m_newfile) throw std::runtime_error("cannot read uninitialized t3 binary file");

      const ca::typeinfo& type = buffer.type();

      if (!buffer.type().is_compatible(m_type_arrayset)) buffer.set(m_type_arrayset);

      //open the file, now for reading the contents...
      std::ifstream ifile(m_filename.c_str(), std::ios::binary|std::ios::in);

      //skip the first 8 bytes, that contain the header that we already read
      ifile.seekg(8 + (index*type.buffer_size()));
      ifile.read(static_cast<char*>(buffer.ptr()), type.buffer_size());

    }

    virtual size_t arrayset_append (const ca::interface& buffer) {

      const ca::typeinfo& info = buffer.type();

      if (!m_newfile && !info.is_compatible(m_type_arrayset)) 
        throw std::invalid_argument("input buffer does not conform to already initialized torch3vision binary file");

      std::ofstream ofile;
      if (m_newfile) {
        
        //can only save uni-dimensional data, so throw if that is not the case
        if (info.nd != 1) throw io::DimensionError(info.nd, 1);

        //can only save float32 or float64, otherwise, throw.
        if ((info.dtype != Torch::core::array::t_float32) && 
            (info.dtype != Torch::core::array::t_float64)) {
          throw io::UnsupportedTypeError(info.dtype);
        }
        
        ofile.open(m_filename.c_str(), std::ios::binary|std::ios::out|std::ios::trunc);

        //header writing...
        const uint32_t nsamples = 0;
        const uint32_t framesize = info.shape[0];
        ofile.write((const char*)&nsamples, sizeof(uint32_t));
        ofile.write((const char*)&framesize, sizeof(uint32_t));
        m_type_arrayset = info;
        m_type_array.dtype = info.dtype;
        m_newfile = false; ///< block re-initialization
        m_length = 0;
        
      }
      else {
        //only open the file, the rest is setup already
        ofile.open(m_filename.c_str(), std::ios::binary|std::ios::out|std::ios::app);
      }

      if (!ofile) throw std::runtime_error("cannot open output file for writing");

      ofile.write(static_cast<const char*>(buffer.ptr()), info.buffer_size());

      //setup new type information
      ++m_length;
      size_t shape[2] = {m_length, info.shape[0]};
      m_type_array.set_shape<size_t>(2, &shape[0]);

      return m_length-1;
      
    }

    virtual void array_write (const ca::interface& buffer) {

      m_newfile = true; //force file re-setting
      arrayset_append(buffer);

    }

  private: //representation

    std::string m_filename;
    bool m_newfile;
    ca::typeinfo m_type_array;
    ca::typeinfo m_type_arrayset;
    size_t m_length;

    static std::string s_codecname;

};

std::string T3File::s_codecname = "torch3.binary";

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

  return boost::make_shared<T3File>(path, mode);

}

/**
 * Takes care of codec registration per se.
 */
static bool register_codec() {

  boost::shared_ptr<io::CodecRegistry> instance =
    io::CodecRegistry::instance();
  
  instance->registerExtension(".bindata", &make_file);

  return true;

}

static bool codec_registered = register_codec();
