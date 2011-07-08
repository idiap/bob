/**
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Implements an image format reader/writer using ImageMagick.
 *
 * This codec will only be able to work with 2D and 3D input.
 */

#include "io/ImageArrayCodec.h"
#include "io/ArrayCodecRegistry.h"
#include "io/Exception.h"
#include <fstream>
#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>
#include <unistd.h>

#include <ImageMagick/Magick++.h> 
#include <ImageMagick/magick/MagickCore.h>

#include <iostream>

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

void io::ImageArrayCodec::peek(const std::string& filename, 
    Torch::core::array::ElementType& eltype, size_t& ndim,
    size_t* shape) const 
{
  try {
    // Read a file into image object
    // TODO: Does ping() retrieve the colorspace and the depth?
    Magick::Image image(filename.c_str());
  
    // Assume Grayscale image
    if( !image.magick().compare("PBM") || !image.magick().compare("PGM") ||
        (!image.magick().compare("PNM") && (image.type() == Magick::BilevelType ||
        image.type() == Magick::GrayscaleMatteType || image.type() == Magick::GrayscaleType) ))
    {
      image.colorSpace( Magick::GRAYColorspace);
      ndim = 2;
      shape[0] = image.rows();
      shape[1] = image.columns();
    // Assume RGB image
    } else {
      image.colorSpace( Magick::RGBColorspace);
      ndim = 3;
      shape[0] = 3;
      shape[1] = image.rows();
      shape[2] = image.columns();
    }

    // Set depth
    if( image.depth() <= 8)
      eltype = Torch::core::array::t_uint8;
    else if( image.depth() <= 16)
      eltype = Torch::core::array::t_uint16;
    else {
      throw io::ImageUnsupportedDepth(image.depth());
    }
  }
  catch( Magick::Exception &error_ ) {
    throw io::FileNotReadable(filename);
  }
}

io::detail::InlinedArrayImpl 
io::ImageArrayCodec::load(const std::string& filename) const {
  try {
    // Read a file into image object
    // TODO: Does ping() retrieve the colorspace and the depth?
    Magick::Image image(filename.c_str());
    // Declare variables
    int ndim;
    int n_c;
    int height;
    int width;
    Torch::core::array::ElementType eltype;
  
    // Assume Grayscale image
    if( !image.magick().compare("PBM") || !image.magick().compare("PGM") ||
        (!image.magick().compare("PNM") && (image.type() == Magick::BilevelType ||
        image.type() == Magick::GrayscaleMatteType || image.type() == Magick::GrayscaleType) ))
    {
      image.colorSpace( Magick::GRAYColorspace);
      ndim = 2;
      n_c = 1;
      height = image.rows();
      width = image.columns();
    // Assume RGB image
    } else {
      image.colorSpace( Magick::RGBColorspace);
      ndim = 3;
      n_c = 3;
      height = image.rows();
      width = image.columns();
    }
    // Set depth
    if( image.depth() <= 8)
      eltype = Torch::core::array::t_uint8;
    else if( image.depth() <= 16)
      eltype = Torch::core::array::t_uint16;
    else {
      throw io::ImageUnsupportedDepth(image.depth());
    }

    // Read the data
    if( eltype == Torch::core::array::t_uint8) {
      // Grayscale
      if( ndim == 2) {
        blitz::Array<uint8_t,2> data( height, width);
        uint8_t *pixels = new uint8_t[width*height];
        image.write( 0, 0, width, height, "I", Magick::CharPixel, pixels );
        for (int h=0; h<height; ++h)
          for (int w=0; w<width; ++w)
            data(h,w) = pixels[h*width+w]; 
        delete [] pixels;
        return io::detail::InlinedArrayImpl(data);
      // RGB
      } else if( ndim == 3) {
        blitz::Array<uint8_t,3> data( n_c, height, width);
        uint8_t *pixels = new uint8_t[n_c*width*height];
        image.write( 0, 0, width, height, "RGB", Magick::CharPixel, pixels );
        for (int h=0; h<height; ++h)
          for (int w=0; w<width; ++w)
            for (int c=0; c<n_c; ++c)
              data(c,h,w) = pixels[n_c*h*width+w*n_c+c]; 
        delete [] pixels;
        return io::detail::InlinedArrayImpl(data);
      }
      else
        throw io::ImageUnsupportedDimension(ndim);
    } else if( eltype == Torch::core::array::t_uint16) {
      // Grayscale
      if( ndim == 2) {
        blitz::Array<uint16_t,2> data( height, width);
        uint16_t *pixels = new uint16_t[width*height];
        image.write( 0, 0, width, height, "I", Magick::ShortPixel, pixels );
        for (int h=0; h<height; ++h)
          for (int w=0; w<width; ++w)
            data(h,w) = pixels[h*width+w]; 
        delete [] pixels;
        return io::detail::InlinedArrayImpl(data);
      // RGB
      } else if( ndim == 3) {
        blitz::Array<uint16_t,3> data( n_c, height, width);
        uint16_t *pixels = new uint16_t[n_c*width*height];
        image.write( 0, 0, width, height, "RGB", Magick::ShortPixel, pixels );
        for (int h=0; h<height; ++h)
          for (int w=0; w<width; ++w)
            for (int c=0; c<n_c; ++c)
              data(c,h,w) = pixels[n_c*h*width+w*n_c+c]; 
        delete [] pixels;
        return io::detail::InlinedArrayImpl(data);
      }
      else
        throw io::ImageUnsupportedDimension(ndim);
    }
    else
      throw io::ImageUnsupportedType(eltype);
  }
  catch( Magick::Exception &error_ ) {
    throw io::FileNotReadable(filename);
  }
}

void io::ImageArrayCodec::save (const std::string& filename,
    const io::detail::InlinedArrayImpl& data) const {
  // Declare variables
  int n_c = 1;
  int ndim;
  int height;
  int width;

  // Get the shape of the array
  const size_t *shape = data.getShape();

  // Save two-dimensional (Grayscale) or tree-dimensional (RGB) array
  // Grayscale
  if(data.getNDim() == 2) {
    ndim = 2;
    height = shape[0];
    width = shape[1];
  // RGB
  } else if( data.getNDim() == 3) {
    ndim = 3;
    n_c = 3;
    // Accept only 3 color channels (RGB)
    if( shape[0] != 3)
      throw io::ImageUnsupportedDimension(ndim);
    height = shape[1];
    width = shape[2];
  }
  // Throw an exception if not supported
  else {
    throw io::ImageUnsupportedDimension(data.getNDim());
  }

  // Create an ImageMagick image
  Magick::Image image;
  image.size( Magick::Geometry( width, height) );
  // Save array/image to file
  if( data.getElementType() == Torch::core::array::t_uint8) {
    // Grayscale
    if( ndim == 2) {
      blitz::Array<uint8_t,2> img = data.get<uint8_t,2>();
      image.colorSpace( Magick::GRAYColorspace);
      uint8_t *pixels = new uint8_t[width*height];
      for (int h=0; h<height; ++h)
        for (int w=0; w<width; ++w)
          pixels[h*width+w] = img(h,w);
      image.read( width, height, "I", Magick::CharPixel, pixels );
      image.depth(8);
      image.write( filename.c_str());
      delete [] pixels;
    }
    // RGB
    else if( ndim == 3) {
      blitz::Array<uint8_t,3> img = data.get<uint8_t,3>();
      image.colorSpace( Magick::RGBColorspace);
      uint8_t *pixels = new uint8_t[width*height*n_c];
      for (int h=0; h<height; ++h)
        for (int w=0; w<width; ++w)
          for (int c=0; c<n_c; ++c)
            pixels[h*width*n_c+w*n_c+c] = img(c,h,w);
      image.read( width, height, "RGB", Magick::CharPixel, pixels );
      image.depth(8);
      image.write( filename.c_str());
      delete [] pixels;
    }
    else {
      throw io::ImageUnsupportedDimension(ndim);
    }
  } else if( data.getElementType() == Torch::core::array::t_uint16) {
    // Grayscale
    if( ndim == 2) {
      blitz::Array<uint16_t,2> img = data.get<uint16_t,2>();
      image.colorSpace( Magick::GRAYColorspace);
      uint16_t *pixels = new uint16_t[width*height];
      for (int h=0; h<height; ++h)
        for (int w=0; w<width; ++w)
          pixels[h*width+w] = img(h,w);
      image.read( width, height, "I", Magick::ShortPixel, pixels );
      image.depth(16);
      image.write( filename.c_str());
      delete [] pixels;
    }
    // RGB
    else if( ndim == 3) {
      blitz::Array<uint16_t,3> img = data.get<uint16_t,3>();
      image.colorSpace( Magick::RGBColorspace);
      uint16_t *pixels = new uint16_t[width*height*n_c];
      for (int h=0; h<height; ++h)
        for (int w=0; w<width; ++w)
          for (int c=0; c<n_c; ++c)
            pixels[h*width*n_c+w*n_c+c] = img(c,h,w);
      image.read( width, height, "RGB", Magick::CharPixel, pixels );
      image.depth(16);
      image.write( filename.c_str());
      delete [] pixels;
    }
    else {
      throw io::ImageUnsupportedDimension(ndim);
    }
  }
  else {
    throw io::ImageUnsupportedType(data.getElementType());
  }
}
