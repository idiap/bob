/**
 * @file io/cxx/ImageFile.cc
 * @date Tue Oct 25 23:25:46 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Implements an image format reader/writer using ImageMagick.
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

#include <ImageMagick/Magick++.h> 

#include "bob/io/CodecRegistry.h"
#include "bob/io/Exception.h"

namespace fs = boost::filesystem;
namespace io = bob::io;
namespace ca = bob::core::array;

static void im_peek(const std::string& path, ca::typeinfo& info) {

  Magick::Image image;
  image.ping(path.c_str());
  std::string ext = boost::filesystem::path(path).extension().c_str();
  boost::algorithm::to_lower(ext);

  if( (!image.magick().compare("PBM") || 
        !image.magick().compare("PGM") ||
        (
         !image.magick().compare("PNM") && 
         (
          image.type() == Magick::BilevelType ||
          image.type() == Magick::GrayscaleMatteType || 
          image.type() == Magick::GrayscaleType) 
        )
      ) && ext != ".ppm" //hack to get around ImageMagic-6.6.x
    )
  {
    // Assume Grayscale image
    info.nd = 2;
    info.shape[0] = image.rows();
    info.shape[1] = image.columns();
    info.update_strides();
  } 
  else {
    // Assume RGB image
    info.nd = 3;
    info.shape[0] = 3;
    info.shape[1] = image.rows();
    info.shape[2] = image.columns();
    info.update_strides();
  }

  // Set depth
  if (image.depth() <= 8) info.dtype = bob::core::array::t_uint8;
  else if (image.depth() <= 16) info.dtype = bob::core::array::t_uint16;
  else {
    boost::format m("unsupported image depth (%d) when reading file");
    m % image.depth();
    throw std::runtime_error(m.str());
  }
}

template <typename T>
Magick::StorageType magick_storage_type() {
  throw std::runtime_error("unknown Image Magick++ storage type -- debug me");
}

template <> Magick::StorageType magick_storage_type<uint8_t>() {
  return Magick::CharPixel;
}

template <> Magick::StorageType magick_storage_type<uint16_t>() {
  return Magick::ShortPixel;
}

template <typename T>
static void im_save_gray(const ca::interface& b, const std::string& name) {
  const ca::typeinfo& info = b.type();

  Magick::Image image;
  image.size(Magick::Geometry(info.shape[1], info.shape[0]));
  image.colorSpace(Magick::GRAYColorspace);
  image.read(info.shape[1], info.shape[0], "I", magick_storage_type<T>(),
      static_cast<const T*>(b.ptr()));
  image.depth(8*sizeof(T));
  image.write(name.c_str());
}

template <typename T> static
void rgb_to_imbuffer(size_t size, const T* r, const T* g, const T* b, T* im) {
  for (size_t k=0; k<size; ++k) {
    im[3*k]    = r[k];
    im[3*k +1] = g[k];
    im[3*k +2] = b[k];
  }
}

template <typename T>
static void im_save_color(const ca::interface& b, const std::string& name) {
  const ca::typeinfo& info = b.type();

  Magick::Image image;
  image.size(Magick::Geometry(info.shape[2], info.shape[1]));
  image.colorSpace(Magick::RGBColorspace);
  long unsigned int frame_size = info.shape[2] * info.shape[1];
  const T* red = static_cast<const T*>(b.ptr());
  const T* green = red + frame_size;
  const T* blue = green + frame_size;
  boost::shared_array<T> tmp(new T[3*frame_size]);
  rgb_to_imbuffer(frame_size, red, green, blue, tmp.get());
  image.read(info.shape[2], info.shape[1], "RGB", 
      magick_storage_type<T>(), tmp.get());
  image.depth(8*sizeof(T));
  image.write(name.c_str());
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
void im_load_gray(Magick::Image& image, ca::interface& b) {
  const ca::typeinfo& info = b.type();

  image.write(0, 0, info.shape[1], info.shape[0], "I", 
      magick_storage_type<T>(), static_cast<T*>(b.ptr()));
}

template <typename T> static
void im_load_color(Magick::Image& image, ca::interface& b) {
  const ca::typeinfo& info = b.type();

  long unsigned int frame_size = info.shape[2] * info.shape[1];
  boost::shared_array<T> tmp(new T[3*frame_size]);
  image.write(0, 0, info.shape[2], info.shape[1], "RGB", 
      magick_storage_type<T>(), tmp.get());
  T* red   = static_cast<T*>(b.ptr());
  T* green = red + frame_size;
  T* blue  = green + frame_size;
  imbuffer_to_rgb(frame_size, tmp.get(), red, green, blue);
}

/**
 * Reads the data.
 */
static void im_load (Magick::Image& image, ca::interface& b) {

  const ca::typeinfo& info = b.type();

  if (info.dtype == bob::core::array::t_uint8) {
    if(info.nd == 2) im_load_gray<uint8_t>(image, b);
    else if( info.nd == 3) im_load_color<uint8_t>(image, b); 
    else throw io::ImageUnsupportedDimension(info.nd);
  }

  else if (info.dtype == bob::core::array::t_uint16) {
    if(info.nd == 2) im_load_gray<uint16_t>(image, b);
    else if( info.nd == 3) im_load_color<uint16_t>(image, b); 
    else throw io::ImageUnsupportedDimension(info.nd);
  }

  else throw io::ImageUnsupportedType(info.dtype);
}

static void im_save (const std::string& filename, const ca::interface& array) {

  const ca::typeinfo& info = array.type();

  if(info.dtype == bob::core::array::t_uint8) {

    if(info.nd == 2) im_save_gray<uint8_t>(array, filename);
    else if(info.nd == 3) {
      if(info.shape[0] != 3) throw std::runtime_error("color image does not have 3 planes on 1st. dimension");
      im_save_color<uint8_t>(array, filename);
    }
    else throw io::ImageUnsupportedDimension(info.nd);

  }

  else if(info.dtype == bob::core::array::t_uint16) {

    if(info.nd == 2) im_save_gray<uint16_t>(array, filename);
    else if(info.nd == 3) {
      if(info.shape[0] != 3) throw std::runtime_error("color image does not have 3 planes on 1st. dimension");
      im_save_color<uint16_t>(array, filename);
    }
    else throw io::ImageUnsupportedDimension(info.nd);

  }

  else throw io::ImageUnsupportedType(info.dtype);
}

class ImageFile: public io::File {

  public: //api

    ImageFile(const std::string& path, char mode):
      m_filename(path),
      m_newfile(true) {

        if (mode == 'r' || (mode == 'a' && fs::exists(path))) { //try peeking
          try {
            im_peek(path, m_type);
            m_length = 1;
            m_newfile = false;
          }
          catch (Magick::Exception &error_) {
            boost::format m("file '%s' is not readable; ImageMagick-%s reports: %s");
            m % path % MagickLibVersionText % error_.what();
            throw std::runtime_error(m.str());
          }
        }
        else {
          m_length = 0;
          m_newfile = true;
        }

      }

    virtual ~ImageFile() { }

    virtual const std::string& filename() const {
      return m_filename;
    }

    virtual const ca::typeinfo& array_type() const {
      return m_type;
    }

    virtual const ca::typeinfo& arrayset_type() const {
      return m_type;
    }

    virtual size_t arrayset_size() const {
      return m_length;
    }

    virtual const std::string& name() const {
      return s_codecname;
    }

    virtual void array_read(ca::interface& buffer) {
      arrayset_read(buffer, 0); ///we only have 1 image in an image file anyways
    }

    virtual void arrayset_read(ca::interface& buffer, size_t index) {

      if (m_newfile) 
        throw std::runtime_error("uninitialized image file cannot be read");

      if (!buffer.type().is_compatible(m_type)) buffer.set(m_type);

      if (index != 0)
        throw std::runtime_error("cannot read image with index > 0 -- there is only one image in an image file");

      try {
        Magick::Image image(m_filename);
        if(!buffer.type().is_compatible(m_type)) buffer.set(m_type);
        im_load(image, buffer);
      }
      catch( Magick::Exception &error_ ) {
        boost::format m("file '%s' is not readable; ImageMagick-%s reports: %s");
        m % m_filename % MagickLibVersionText % error_.what();
        throw std::runtime_error(m.str());
      }

    }

    virtual size_t arrayset_append (const ca::interface& buffer) {

      if (m_newfile) {
        im_save(m_filename, buffer);
        m_type = buffer.type();
        m_newfile = false;
        m_length = 1;
        return 0;
      }

      throw std::runtime_error("image files only accept a single array");

    }

    virtual void array_write (const ca::interface& buffer) {

      //overwriting position 0 should always work
      if (m_newfile) {
        arrayset_append(buffer);
        return;
      }

      throw std::runtime_error("image files only accept a single array");
    }

  private: //representation
    std::string m_filename;
    bool m_newfile;
    ca::typeinfo m_type;
    size_t m_length;

    static std::string s_codecname;

};

std::string ImageFile::s_codecname = "bob.image";

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

  return boost::make_shared<ImageFile>(path, mode);

}

/**
 * Takes care of codec registration per se.
 */
static bool register_codec() {

  boost::shared_ptr<io::CodecRegistry> instance =
    io::CodecRegistry::instance();
  
  instance->registerExtension(".bmp", "Windows bitmap (Image Magick)", &make_file);
  instance->registerExtension(".eps", "Encapsulated Postscript (Image Magick)", &make_file);
  instance->registerExtension(".gif", "GIF, indexed (Image Magick)", &make_file);
  instance->registerExtension(".jpg", "JPG, compressed (Image Magick)", &make_file);
  instance->registerExtension(".jpeg", "JPEG, compressed (Image Magick)", &make_file);
  instance->registerExtension(".pbm", "PBM, indexed (Image Magick)", &make_file);
  instance->registerExtension(".pdf", "Portable Document Format (Image Magick)", &make_file);
  instance->registerExtension(".pgm", "PGM, indexed (Image Magick)", &make_file);
  instance->registerExtension(".png", "Portable Network Graphics, indexed (Image Magick)", &make_file);
  instance->registerExtension(".ppm", "PPM, indexed (Image Magick)", &make_file);
  instance->registerExtension(".ps", "Postscript (Image Magick)", &make_file);
  instance->registerExtension(".tiff", "TIFF, uncompressed (Image Magick)", &make_file);
  instance->registerExtension(".xcf", "Gimp Native format (ImageMagick)", &make_file);

  return true;

}

static bool codec_registered = register_codec();
