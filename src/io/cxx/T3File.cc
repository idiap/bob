/**
 * @file io/cxx/T3File.cc
 * @date Wed Oct 26 17:11:16 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Implements a torch3vision bindata reader/writer
 * The format, as described in the old source code goes like this.
 * 1) data is always recorded in little endian format
 * 2) the first 4 bytes describe an integer that indicates the number of arrays
 * to follow
 * 3) the second 4 bytes describe an integer that specifies the frame width.
 * 4) all arrays inserted there are single dimensional arrays.
 * 5) all elements from all arrays are "normally" float (4-bytes), but could be
 * double if set in the header of T3 during compilation. The file size will
 * indicate the right type to use.
 * Because of this restriction, this codec will only be able to work with
 * single-dimension input.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/make_shared.hpp>
#include <boost/format.hpp>

//some infrastructure to check the file size
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <bob/core/check.h>
#include <bob/core/blitz_array.h>
#include <bob/io/CodecRegistry.h>

static inline size_t get_filesize(const std::string& filename) {
  struct stat filestatus;
  stat(filename.c_str(), &filestatus);
  return filestatus.st_size;
}

class T3File: public bob::io::File {

  public: //api

    T3File(const std::string& path, char mode):
      m_filename(path),
      m_newfile(true),
      m_length(0) {
        if ( mode == 'r' || (mode == 'a' && boost::filesystem::exists(path) ) ) { // try peek
          size_t fsize = get_filesize(path);
          fsize -= 8; // remove the first two entries
          // read the first two 4-byte integers in the file, convert to unsigned

          std::fstream s(path.c_str(), std::ios::binary|std::ios::in);

          if (!s) {
            boost::format m("cannot open file `%s'");
            m % path;
            throw std::runtime_error(m.str());
          }

          uint32_t nsamples, framesize;
          nsamples = framesize = 0;
          s.read((char*)&nsamples, sizeof(uint32_t));
          s.read((char*)&framesize, sizeof(uint32_t));

          m_length = nsamples;

          // are those floats or doubles?
          if (fsize == (nsamples*framesize*sizeof(float))) {
            m_type_array.dtype = bob::core::array::t_float32;
            m_type_arrayset.dtype = bob::core::array::t_float32;
          }
          else if (fsize == (nsamples*framesize*sizeof(double))) {
            m_type_array.dtype = bob::core::array::t_float64;
            m_type_arrayset.dtype = bob::core::array::t_float64;
          }
          else {
            boost::format s("Cannot read file '%s', mode = '%c': fsize (%d) != %d*%d*sizeof(float32) nor *sizeof(float64)");
            s % path % mode % fsize % nsamples % framesize;
            throw std::invalid_argument(s.str());
          }

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

    virtual const bob::core::array::typeinfo& type_all () const {
      return m_type_array;
    }

    virtual const bob::core::array::typeinfo& type () const {
      return m_type_arrayset;
    }

    virtual size_t size() const {
      return m_length;
    }

    virtual const std::string& name() const {
      return s_codecname;
    }

    virtual void read_all(bob::core::array::interface& buffer) {

      if (m_newfile) {
        boost::format f("cannot read uninitialized t3 binary file at '%s'");
        f % m_filename;
        throw std::runtime_error(f.str());
      }

      if (!buffer.type().is_compatible(m_type_array)) buffer.set(m_type_array);

      //open the file, now for reading the contents...
      std::ifstream ifile(m_filename.c_str(), std::ios::binary|std::ios::in);

      //skip the first 8 bytes, that contain the header that we already read
      ifile.seekg(8, std::ios::beg);
      ifile.read(static_cast<char*>(buffer.ptr()), buffer.type().buffer_size());

    }

    virtual void read(bob::core::array::interface& buffer, size_t index) {

      if (m_newfile) {
        boost::format f("cannot read uninitialized t3 binary file at '%s'");
        f % m_filename;
        throw std::runtime_error(f.str());
      }

      const bob::core::array::typeinfo& type = buffer.type();

      if (!buffer.type().is_compatible(m_type_arrayset)) buffer.set(m_type_arrayset);

      //open the file, now for reading the contents...
      std::ifstream ifile(m_filename.c_str(), std::ios::binary|std::ios::in);

      //skip the first 8 bytes, that contain the header that we already read
      ifile.seekg(8 + (index*type.buffer_size()), std::ios::beg);
      ifile.read(static_cast<char*>(buffer.ptr()), type.buffer_size());

    }

    virtual size_t append (const bob::core::array::interface& buffer) {

      const bob::core::array::typeinfo& info = buffer.type();

      if (!m_newfile && !info.is_compatible(m_type_arrayset)) {
        boost::format f("input buffer of type %s cannot be appended to already initialized torch3vision binary file of type %s");
        f % info.str() % m_type_arrayset.str();
        throw std::invalid_argument(f.str());
      }

      std::ofstream ofile;
      if (m_newfile) {

        //can only save uni-dimensional data, so throw if that is not the case
        if (info.nd != 1) {
          boost::format m("codec for torch3vision binary files can only save uni-dimensional data, but you passed: %s");
          m % info.str();
          throw std::runtime_error(m.str());
        }

        //can only save float32 or float64, otherwise, throw.
        if ((info.dtype != bob::core::array::t_float32) &&
            (info.dtype != bob::core::array::t_float64)) {
          boost::format f("cannot have T3 bindata files with type %s - only float32 or float64");
          f % bob::core::array::stringize(info.dtype);
          throw std::invalid_argument(f.str());
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

      if (!ofile) {
        boost::format f("cannot open output file '%s' for writing");
        f % m_filename;
        throw std::runtime_error(f.str());
      }

      ofile.write(static_cast<const char*>(buffer.ptr()), info.buffer_size());
      ofile.close();

      //setup new type information
      ++m_length;
      size_t shape[2] = {m_length, info.shape[0]};
      m_type_array.set_shape<size_t>(2, &shape[0]);

      //update the header information on the file
      ofile.open(m_filename.c_str(), std::ios::binary|std::ios::in|std::ios::out);
      const uint32_t nsamples = m_length;
      ofile.write((const char*)&nsamples, sizeof(uint32_t));
      ofile.flush();
      return m_length-1;

    }

    /**
     * Supports writing a single vector or a set of vectors represented as a
     * matrix. In this last case, vectors are formed from the rows of the given
     * matrix.
     */
    virtual void write (const bob::core::array::interface& buffer) {

      m_newfile = true; //force file re-setting
      const bob::core::array::typeinfo& info = buffer.type();

      if (info.nd == 1) {//just do a normal append
        append(buffer);
      }

      else if (info.nd == 2) { //append every array individually

        const uint8_t* ptr = static_cast<const uint8_t*>(buffer.ptr());
        bob::core::array::typeinfo slice_info(info.dtype, static_cast<size_t>(1),
            &info.shape[1]);
        for (size_t k=0; k<info.shape[0]; ++k) {
          const void* slice_ptr=static_cast<const void*>(ptr+k*slice_info.buffer_size());
          bob::core::array::blitz_array slice(const_cast<void*>(slice_ptr), slice_info);
          append(slice);
        }

      }

      else {
        boost::format f("cannot do single write of torch3vision .bindata file with array with type '%s' - only supports 1D or 2D arrays of types float32 or float64");
        f % info.str();
        throw std::invalid_argument(f.str());
      }

    }

  private: //representation

    std::string m_filename;
    bool m_newfile;
    bob::core::array::typeinfo m_type_array;
    bob::core::array::typeinfo m_type_arrayset;
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
static boost::shared_ptr<bob::io::File>
make_file (const std::string& path, char mode) {

  return boost::make_shared<T3File>(path, mode);

}

/**
 * Takes care of codec registration per se.
 */
static bool register_codec() {

  boost::shared_ptr<bob::io::CodecRegistry> instance =
    bob::io::CodecRegistry::instance();

  instance->registerExtension(".bindata", "torch3 binary data format", &make_file);

  return true;

}

static bool codec_registered = register_codec();
