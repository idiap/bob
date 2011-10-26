/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @author Laurent El-Shafey <laurent.el-shafey@idiap.ch>
 * @date Thu  6 Oct 16:22:33 2011 CEST
 *
 * @brief Implements an image format reader/writer using ImageMagick.
 *
 * This codec is only able to work with 2D and 3D input.
 */

#include <boost/filesystem.hpp>
#include <boost/shared_array.hpp>
#include <boost/make_shared.hpp>

#include <ImageMagick/Magick++.h> 
#include <ImageMagick/magick/MagickCore.h>

#include "io/CodecRegistry.h"
#include "io/Exception.h"

namespace fs = boost::filesystem;
namespace io = Torch::io;

static int im_peek(Magick::Image& image, io::typeinfo& info) {
  int retval = 0;

    // Assume Grayscale image
    if( !image.magick().compare("PBM") || 
        !image.magick().compare("PGM") ||
        (
         !image.magick().compare("PNM") && 
         (
          image.type() == Magick::BilevelType ||
          image.type() == Magick::GrayscaleMatteType || 
          image.type() == Magick::GrayscaleType) 
         )
        )
    {
      image.colorSpace(Magick::GRAYColorspace);
      info.nd = 2;
      info.shape[0] = image.rows();
      info.shape[1] = image.columns();
      info.update_strides();
      retval = 1;
    } 
    else {
    // Assume RGB image
      image.colorSpace(Magick::RGBColorspace);
      info.nd = 3;
      info.shape[0] = 3;
      info.shape[1] = image.rows();
      info.shape[2] = image.columns();
      info.update_strides();
      retval = 3;
    }

    // Set depth
    if (image.depth() <= 8) info.dtype = Torch::core::array::t_uint8;
    else if (image.depth() <= 16) info.dtype = Torch::core::array::t_uint16;
    else throw std::runtime_error("unsupported image depth when reading file");

  return retval;
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
static void im_save_gray(const io::buffer& b, const std::string& name) {
  const io::typeinfo& info = b.type();

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
static void im_save_color(const io::buffer& b, const std::string& name) {
  const io::typeinfo& info = b.type();

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
void im_load_gray(Magick::Image& image, io::buffer& b) {
  const io::typeinfo& info = b.type();

  image.write(0, 0, info.shape[1], info.shape[0], "I", 
      magick_storage_type<T>(), static_cast<T*>(b.ptr()));
}

template <typename T> static
void im_load_color(Magick::Image& image, io::buffer& b) {
  const io::typeinfo& info = b.type();

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
static void im_load (Magick::Image& image, io::buffer& b) {

  const io::typeinfo& info = b.type();

  if (info.dtype == Torch::core::array::t_uint8) {
    if(info.nd == 2) im_load_gray<uint8_t>(image, b);
    else if( info.nd == 3) im_load_color<uint8_t>(image, b); 
    else throw io::ImageUnsupportedDimension(info.nd);
  }

  else if (info.dtype == Torch::core::array::t_uint16) {
    if(info.nd == 2) im_load_gray<uint16_t>(image, b);
    else if( info.nd == 3) im_load_color<uint16_t>(image, b); 
    else throw io::ImageUnsupportedDimension(info.nd);
  }

  else throw io::ImageUnsupportedType(info.dtype);
}

static void im_save (const std::string& filename, const io::buffer& array) {

  const io::typeinfo& info = array.type();

  if(info.dtype == Torch::core::array::t_uint8) {

    if(info.nd == 2) im_save_gray<uint8_t>(array, filename);
    else if(info.nd == 3) {
      if(info.shape[0] != 3) throw std::runtime_error("color image does not have 3 planes on 1st. dimension");
      im_save_color<uint8_t>(array, filename);
    }
    else throw io::ImageUnsupportedDimension(info.nd);

  }

  else if(info.dtype == Torch::core::array::t_uint16) {

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
            Magick::Image image;
            image.ping(path.c_str());
            im_peek(image, m_type);
            m_length = 1;
            m_newfile = false;
          }
          catch (Magick::Exception &error_) {
            throw io::FileNotReadable(path);
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

    virtual const io::typeinfo& array_type() const {
      return m_type;
    }

    virtual const io::typeinfo& arrayset_type() const {
      return m_type;
    }

    virtual size_t arrayset_size() const {
      return m_length;
    }

    virtual const std::string& name() const {
      return s_codecname;
    }

    virtual void array_read(io::buffer& buffer) {
      arrayset_read(buffer, 0); ///we only have 1 image in an image file anyways
    }

    virtual void arrayset_read(io::buffer& buffer, size_t index) {

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
        throw io::FileNotReadable(m_filename);
      }

    }

    virtual size_t arrayset_append (const io::buffer& buffer) {

      if (m_newfile) {
        im_save(m_filename, buffer);
        m_type = buffer.type();
        m_newfile = false;
        m_length = 1;
        return 0;
      }

      throw std::runtime_error("image files only accept a single array");

    }

    virtual void array_write (const io::buffer& buffer) {

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
    io::typeinfo m_type;
    size_t m_length;

    static std::string s_codecname;

};

std::string ImageFile::s_codecname = "torch.image";

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
  
  instance->registerExtension(".bmp", &make_file);
  instance->registerExtension(".eps", &make_file);
  instance->registerExtension(".gif", &make_file);
  instance->registerExtension(".jpg", &make_file);
  instance->registerExtension(".jpeg", &make_file);
  instance->registerExtension(".pbm", &make_file);
  instance->registerExtension(".pdf", &make_file);
  instance->registerExtension(".pgm", &make_file);
  instance->registerExtension(".png", &make_file);
  instance->registerExtension(".ppm", &make_file);
  instance->registerExtension(".ps", &make_file);
  instance->registerExtension(".tiff", &make_file);
  instance->registerExtension(".xcf", &make_file);

  return true;

}

static bool codec_registered = register_codec();
