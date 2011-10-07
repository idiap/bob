/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @author Laurent El-Shafey <laurent.el-shafey@idiap.ch>
 * @date Thu  6 Oct 16:22:33 2011 CEST
 *
 * @brief Implements an image format reader/writer using ImageMagick.
 *
 * This codec is only able to work with 2D and 3D input.
 */

#include <boost/shared_array.hpp>

#include <ImageMagick/Magick++.h> 
#include <ImageMagick/magick/MagickCore.h>

#include "io/ImageArrayCodec.h"
#include "io/ArrayCodecRegistry.h"
#include "io/Exception.h"

namespace io = Torch::io;

//Takes care of the codec registration.
static bool register_codec() {
  io::ArrayCodecRegistry::addCodec(boost::shared_ptr<io::ArrayCodec>(new io::ImageArrayCodec())); 
  return true;
}

static bool codec_registered = register_codec();

io::ImageArrayCodec::ImageArrayCodec()
  : m_name("torch.image"),
    m_extensions()
{ 
  m_extensions.push_back(".bmp");
  m_extensions.push_back(".eps");
  m_extensions.push_back(".gif");
  m_extensions.push_back(".jpg");
  m_extensions.push_back(".jpeg");
  m_extensions.push_back(".pbm");
  m_extensions.push_back(".pdf");
  m_extensions.push_back(".pgm");
  m_extensions.push_back(".png");
  m_extensions.push_back(".ppm");
  m_extensions.push_back(".ps");
  m_extensions.push_back(".tiff");
  m_extensions.push_back(".xcf");
}

io::ImageArrayCodec::~ImageArrayCodec() { }

static int im_peek(Magick::Image& image, io::typeinfo& info) {
  int retval = 0;

    // Assume Grayscale image
    if( !image.magick().compare("PBM") || !image.magick().compare("PGM") ||
        (!image.magick().compare("PNM") && (image.type() == Magick::BilevelType ||
        image.type() == Magick::GrayscaleMatteType || image.type() == Magick::GrayscaleType) ))
    {
      image.colorSpace( Magick::GRAYColorspace);
      info.nd = 2;
      info.shape[0] = image.rows();
      info.shape[1] = image.columns();
      info.update_strides();
      retval = 1;
    // Assume RGB image
    } else {
      image.colorSpace( Magick::RGBColorspace);
      info.nd = 3;
      info.shape[0] = 3;
      info.shape[1] = image.rows();
      info.shape[2] = image.columns();
      info.update_strides();
      retval = 3;
    }

    // Set depth
    if( image.depth() <= 8)
      info.dtype = Torch::core::array::t_uint8;
    else if( image.depth() <= 16)
      info.dtype = Torch::core::array::t_uint16;
    else {
      throw io::ImageUnsupportedDepth(image.depth());
    }
  return retval;
}

void io::ImageArrayCodec::peek(const std::string& filename, io::typeinfo& info) const {
  try {
    Magick::Image image;
    image.ping(filename.c_str());
    im_peek(image, info);
  }
  catch(Magick::Exception &error_ ) {
    throw io::FileNotReadable(filename);
  }
}

template <typename T> static
void imbuffer_to_rgb(size_t size, const T* im, T* r, T* g, T* b) {
  for (size_t k=0; k<size; ++k) {
    r[k] = im[3*k];
    g[k] = im[3*k +1];
    b[k] = im[3*k +2];
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
  boost::shared_array<T> tmp(new T[frame_size]);
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
static void im_load(Magick::Image& image, io::buffer& b) {
  
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

void io::ImageArrayCodec::load(const std::string& filename, io::buffer& array) const {
  try {
    io::typeinfo info;
    Magick::Image image(filename.c_str());
    im_peek(image, info);
    if(!array.type().is_compatible(info)) array.set(info);
    im_load(image, array);
  }
  catch( Magick::Exception &error_ ) {
    throw io::FileNotReadable(filename);
  }
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
  boost::shared_array<T> tmp(new T[frame_size]);
  rgb_to_imbuffer(frame_size, red, green, blue, tmp.get());
  image.read(info.shape[2], info.shape[1], "RGB", 
      magick_storage_type<T>(), tmp.get());
  image.depth(8*sizeof(T));
  image.write(name.c_str());
}

void io::ImageArrayCodec::save (const std::string& filename, 
    const io::buffer& array) const {
  
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
