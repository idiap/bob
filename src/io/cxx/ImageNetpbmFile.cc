/**
 * @file io/cxx/ImageNetpbmFile.cc
 * @date Tue Oct 9 18:13:00 2012 +0200
 * @author Laurent El Shafey <laurent.el-shafey@idiap.ch>
 *
 * @brief Implements an image format reader/writer using libnetpbm.
 * This codec is only able to work with 2D and 3D input.
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
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

#include <boost/filesystem.hpp>
#include <boost/shared_array.hpp>
#include <boost/make_shared.hpp>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <string>

#include "bob/io/CodecRegistry.h"
#include "bob/io/Exception.h"

extern "C" {
// This header must come last, as it brings a lot of stuff that
// messes up other headers...
#include <pam.h>
}



static void im_peek(const std::string& path, bob::core::array::typeinfo& info) {

  struct pam in_pam;
  FILE *in_file = pm_openr(path.c_str());
#ifdef PAM_STRUCT_SIZE // For version >= 10.23
  pnm_readpaminit(in_file, &in_pam, PAM_STRUCT_SIZE(tuple_type));
#else
  pnm_readpaminit(in_file, &in_pam, sizeof(struct pam));
#endif
  pm_close(in_file);

  if( in_pam.depth != 1 && in_pam.depth != 3)
  {
    boost::format m("unsupported number of planes (%d) when reading file. Image depth must be 1 or 3.");
    m % in_pam.depth;
    throw std::runtime_error(m.str());
  }

  info.nd = (in_pam.depth == 1? 2 : 3);
  if(info.nd == 2)
  {
    info.shape[0] = in_pam.height;
    info.shape[1] = in_pam.width;
  }
  else
  {
    info.shape[0] = 3;
    info.shape[1] = in_pam.height;
    info.shape[2] = in_pam.width;
  }
  info.update_strides();

  // Set depth
  if (in_pam.bytes_per_sample == 1) info.dtype = bob::core::array::t_uint8;
  else if (in_pam.bytes_per_sample == 2) info.dtype = bob::core::array::t_uint16;
  else {
    boost::format m("unsupported image depth (%d bytes per samples) when reading file");
    m % in_pam.bytes_per_sample;
    throw std::runtime_error(m.str());
  }
}

template <typename T>
static void im_save_gray(const bob::core::array::interface& b, struct pam *out_pam) {
  const bob::core::array::typeinfo& info = b.type();

  const T *element = static_cast<const T*>(b.ptr());

  tuple *tuplerow = pnm_allocpamrow(out_pam);
  for(size_t y=0; y<info.shape[0]; ++y) {
    for(size_t x=0; x<info.shape[1]; ++x) {
      tuplerow[x][0] = *element;
      ++element;
    }
    pnm_writepamrow(out_pam, tuplerow);
  }
  pnm_freepamrow(tuplerow);
}

template <typename T> static
void rgb_to_imbuffer(size_t size, const T* r, const T* g, const T* b, T* im) {
  for (size_t k=0; k<size; ++k) {
    im[3*k]    = r[k];
    im[3*k+1] = g[k];
    im[3*k+2] = b[k];
  }
}

template <typename T>
static void im_save_color(const bob::core::array::interface& b, struct pam *out_pam) {
  const bob::core::array::typeinfo& info = b.type();

  long unsigned int frame_size = info.shape[2] * info.shape[1];
  const T *element_r = static_cast<const T*>(b.ptr());
  const T *element_g = element_r + frame_size;
  const T *element_b = element_g + frame_size;

  tuple *tuplerow = pnm_allocpamrow(out_pam);
  for(size_t y=0; y<info.shape[1]; ++y) {
    for(size_t x=0; x<info.shape[2]; ++x) {
      tuplerow[x][0] = *element_r;
      tuplerow[x][1] = *element_g;
      tuplerow[x][2] = *element_b;
      ++element_r;
      ++element_g;
      ++element_b;
    }
    pnm_writepamrow(out_pam, tuplerow); 
  }
  pnm_freepamrow(tuplerow);
}

template <typename T> static
void imbuffer_to_rgb(size_t size, const T* im, T* r, T* g, T* b) {
  for (size_t k=0; k<size; ++k) {
    r[k] = im[3*k];
    g[k] = im[3*k +1];
    b[k] = im[3*k +2];
  }
}

template <typename T> static
void im_load_gray(const FILE* in_file, struct pam *in_pam, bob::core::array::interface& b) {
  const bob::core::array::typeinfo& info = b.type();

  T *element = static_cast<T*>(b.ptr());
  tuple *tuplerow = pnm_allocpamrow(in_pam);
  for(size_t y=0; y<info.shape[0]; ++y)
  {
    pnm_readpamrow(in_pam, tuplerow);
    for(size_t x=0; x<info.shape[1]; ++x)
    {
      *element = tuplerow[x][0];
      ++element;
    }
  }
  pnm_freepamrow(tuplerow);  
}

template <typename T> static
void im_load_color(const FILE* in_file, struct pam *in_pam, bob::core::array::interface& b) {
  const bob::core::array::typeinfo& info = b.type();
  
  long unsigned int frame_size = info.shape[2] * info.shape[1]; 
  T *element_r = static_cast<T*>(b.ptr());
  T *element_g = element_r+frame_size;
  T *element_b = element_g+frame_size;

  tuple *tuplerow = pnm_allocpamrow(in_pam);
  for(size_t y=0; y<info.shape[1]; ++y)
  {
    pnm_readpamrow(in_pam, tuplerow);
    for(size_t x=0; x<info.shape[2]; ++x)
    {
      *element_r = tuplerow[x][0];
      *element_g = tuplerow[x][1];
      *element_b = tuplerow[x][2];
      ++element_r;
      ++element_g;
      ++element_b;
    }
  }
  pnm_freepamrow(tuplerow);  
}

/**
 * Reads the data.
 */
static void im_load (const std::string& filename, bob::core::array::interface& b) {

  struct pam in_pam;
  FILE *in_file = pm_openr(filename.c_str());
#ifdef PAM_STRUCT_SIZE 
  // For version >= 10.23
  pnm_readpaminit(in_file, &in_pam, PAM_STRUCT_SIZE(tuple_type));
#else
  pnm_readpaminit(in_file, &in_pam, sizeof(struct pam));
#endif

  const bob::core::array::typeinfo& info = b.type();

  if (info.dtype == bob::core::array::t_uint8) {
    if(info.nd == 2) im_load_gray<uint8_t>(in_file, &in_pam, b);
    else if( info.nd == 3) im_load_color<uint8_t>(in_file, &in_pam, b); 
    else { pm_close(in_file); throw bob::io::ImageUnsupportedDimension(info.nd); }
  }

  else if (info.dtype == bob::core::array::t_uint16) {
    if(info.nd == 2) im_load_gray<uint16_t>(in_file, &in_pam, b);
    else if( info.nd == 3) im_load_color<uint16_t>(in_file, &in_pam, b); 
    else { pm_close(in_file); throw bob::io::ImageUnsupportedDimension(info.nd); }
  }

  else { pm_close(in_file); throw bob::io::ImageUnsupportedType(info.dtype); }
  // Close file if no exception
  pm_close(in_file);
}

static void im_save (const std::string& filename, const bob::core::array::interface& array) {

  const bob::core::array::typeinfo& info = array.type();

  struct pam out_pam;
  FILE *out_file = pm_openw(filename.c_str());

  std::string ext = boost::filesystem::path(filename).extension().c_str();
  boost::algorithm::to_lower(ext);

  // Sets the parameters of the pam structure according to the bca::interface properties
  out_pam.size = sizeof(out_pam);
#ifdef PAM_STRUCT_SIZE 
  // For version >= 10.23
  out_pam.len = PAM_STRUCT_SIZE(tuple_type);
#else
  out_pam.len = out_pam.size;
#endif
  out_pam.file = out_file;
  out_pam.plainformat = 0; // writes in binary
  out_pam.height = (info.nd == 2 ? info.shape[0] : info.shape[1]);
  out_pam.width = (info.nd == 2 ? info.shape[1] : info.shape[2]);
  out_pam.depth = (info.nd == 2 ? 1 : 3);
  out_pam.maxval = (bob::core::array::t_uint8 ? 255 : 65535);
  out_pam.bytes_per_sample = (info.dtype == bob::core::array::t_uint8 ? 1 : 2);
  out_pam.format = PAM_FORMAT;
  if( ext.compare(".pbm") == 0) 
  {
    out_pam.maxval = 1;
    strcpy(out_pam.tuple_type, PAM_PBM_TUPLETYPE);
  }
  else if( ext.compare(".pgm") == 0) strcpy(out_pam.tuple_type, PAM_PGM_TUPLETYPE);
  else strcpy(out_pam.tuple_type, PAM_PPM_TUPLETYPE);

  if(out_pam.depth == 3 && ext.compare(".ppm")) {
    pm_close(out_file);
    throw std::runtime_error("cannot save a color image into a file of this type.");
  }

  // Writes header in file
  pnm_writepaminit(&out_pam);

  // Writes content
  if(info.dtype == bob::core::array::t_uint8) {

    if(info.nd == 2) im_save_gray<uint8_t>(array, &out_pam);
    else if(info.nd == 3) {
      if(info.shape[0] != 3) { pm_close(out_file); throw std::runtime_error("color image does not have 3 planes on 1st. dimension");}
      im_save_color<uint8_t>(array, &out_pam);
    }
    else { pm_close(out_file); throw bob::io::ImageUnsupportedDimension(info.nd); }

  }

  else if(info.dtype == bob::core::array::t_uint16) {

    if(info.nd == 2) im_save_gray<uint16_t>(array, &out_pam);
    else if(info.nd == 3) {
      if(info.shape[0] != 3) throw std::runtime_error("color image does not have 3 planes on 1st. dimension");
      im_save_color<uint16_t>(array, &out_pam);
    }
    else { pm_close(out_file); throw bob::io::ImageUnsupportedDimension(info.nd); }

  }

  else { pm_close(out_file); throw bob::io::ImageUnsupportedType(info.dtype); }

  pm_close(out_file);
}


class ImageNetpbmFile: public bob::io::File {

  public: //api

    ImageNetpbmFile(const std::string& path, char mode):
      m_filename(path),
      m_newfile(true) {

        //checks if file exists
        if (mode == 'r' && !boost::filesystem::exists(path)) {
          boost::format m("file '%s' is not readable");
          m % path;
          throw std::runtime_error(m.str());
        }

        if (mode == 'r' || (mode == 'a' && boost::filesystem::exists(path))) {
          {
            im_peek(path, m_type);
            m_length = 1;
            m_newfile = false;
          }
        }
        else {
          m_length = 0;
          m_newfile = true;
        }
      }

    virtual ~ImageNetpbmFile() { }

    virtual const std::string& filename() const {
      return m_filename;
    }

    virtual const bob::core::array::typeinfo& type_all() const {
      return m_type;
    }

    virtual const bob::core::array::typeinfo& type() const {
      return m_type;
    }

    virtual size_t size() const {
      return m_length;
    }

    virtual const std::string& name() const {
      return s_codecname;
    }

    virtual void read_all(bob::core::array::interface& buffer) {
      read(buffer, 0); ///we only have 1 image in an image file anyways
    }

    virtual void read(bob::core::array::interface& buffer, size_t index) {

      if (m_newfile) 
        throw std::runtime_error("uninitialized image file cannot be read");

      if (!buffer.type().is_compatible(m_type)) buffer.set(m_type);

      if (index != 0)
        throw std::runtime_error("cannot read image with index > 0 -- there is only one image in an image file");

      if(!buffer.type().is_compatible(m_type)) buffer.set(m_type);
      im_load(m_filename, buffer);
    }

    virtual size_t append (const bob::core::array::interface& buffer) {
      if (m_newfile) {
        im_save(m_filename, buffer);
        m_type = buffer.type();
        m_newfile = false;
        m_length = 1;
        return 0;
      }

      throw std::runtime_error("image files only accept a single array");
    }

    virtual void write (const bob::core::array::interface& buffer) {
      //overwriting position 0 should always work
      if (m_newfile) {
        append(buffer);
        return;
      }

      throw std::runtime_error("image files only accept a single array");
    }

  private: //representation
    std::string m_filename;
    bool m_newfile;
    bob::core::array::typeinfo m_type;
    size_t m_length;

    static std::string s_codecname;

};

std::string ImageNetpbmFile::s_codecname = "bob.image_netpbm";

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
  return boost::make_shared<ImageNetpbmFile>(path, mode);
}

/**
 * Takes care of codec registration per se.
 */
static bool register_codec() {
  boost::shared_ptr<bob::io::CodecRegistry> instance =
    bob::io::CodecRegistry::instance();

  pm_init("bob",0); 
  instance->registerExtension(".pbm", "PBM, indexed (libnetpbm)", &make_file);
  instance->registerExtension(".pgm", "PGM, indexed (libnetpbm)", &make_file);
  instance->registerExtension(".ppm", "PPM, indexed (libnetpbm)", &make_file);

  return true;

}

static bool codec_registered = register_codec();

