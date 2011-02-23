/**
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Implements an image format reader/writer
 *
 * This codec will only be able to work with three-dimension input.
 */

#include "database/ImageArrayCodec.h"
#include "database/ArrayCodecRegistry.h"
#include "database/Exception.h"
#include <fstream>
#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>
#include <unistd.h>

#include <ImageMagick/Magick++.h> 

namespace db = Torch::database;

namespace Torch { namespace database {
  class ImageException: public Torch::core::Exception { };
}}

//Takes care of the codec registration.
static bool register_codec() {
  db::ArrayCodecRegistry::addCodec(boost::shared_ptr<db::ArrayCodec>(new db::ImageArrayCodec())); 
  return true;
}

static bool codec_registered = register_codec();

db::ImageArrayCodec::ImageArrayCodec()
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

db::ImageArrayCodec::~ImageArrayCodec() { }

void db::ImageArrayCodec::peek(const std::string& filename, 
    Torch::core::array::ElementType& eltype, size_t& ndim,
    size_t* shape) const 
{
  Magick::Image image;
  try {
    // Read a file into image object
    // TODO: Does ping() retrieve the colorspace and the depth?
    image.read(filename.c_str());
  
    if( image.colorSpace() == Magick::GRAYColorspace)
      shape[0] = 1;
    else if( image.colorSpace() == Magick::RGBColorspace)
      shape[0] = 3;
    else // Unsupported colorspace TODO: additional ones?
      throw db::ImageException();
    shape[1] = image.rows();
    shape[2] = image.columns();

    // Set other attributes TODO: depth
    if( image.depth() == 16)
      eltype = Torch::core::array::t_uint16;
    else 
      throw db::ImageException();
    ndim = 3; 
  }
  catch( Magick::Exception &error_ )
  {
    throw db::FileNotReadable(filename);
  } 
}

db::detail::InlinedArrayImpl 
db::ImageArrayCodec::load(const std::string& filename) const {
  Magick::Image image;
  size_t shape[3];
  Torch::core::array::ElementType eltype;
  try {
    // Read a file into image object
    // TODO: Does ping() retrieve the colorspace and the depth?
    image.read(filename.c_str());
  
    if( image.colorSpace() == Magick::GRAYColorspace)
      shape[0] = 1;
    else if( image.colorSpace() == Magick::RGBColorspace)
      shape[0] = 3;
    else // Unsupported colorspace TODO: additional ones?
      throw db::ImageException();
    shape[1] = image.rows();
    shape[2] = image.columns();

    // Set other attributes TODO: depth
    if( image.depth() == 16)
      eltype = Torch::core::array::t_uint16;
    else 
      throw db::ImageException();
  }
  catch( Magick::Exception &error_ )
  {
    throw db::ImageException();
  } 

  // Get the with and the height
  int n_c = shape[0];
  int height = shape[1];
  int width = shape[2];

  if( eltype == Torch::core::array::t_uint16) {
    blitz::Array<uint16_t,3> data(n_c, height, width);
    if(n_c == 1) {
      uint16_t *pixels = new uint16_t[width*height];
      image.write( 0, 0, width, height, "I", Magick::ShortPixel, pixels );
      for (int h=0; h<height; ++h)
        for (int w=0; w<width; ++w)
          data(0,h,w) = pixels[h*width+w]; 
      delete [] pixels;
    }
    else if(n_c == 3) {
      const Magick::PixelPacket *pixels=image.getConstPixels(0,0,width,height);
      for (int h=0; h<height; ++h)
        for (int w=0; w<width; ++w) {
          data(0,h,w) = pixels[h*width+w].red; 
          data(1,h,w) = pixels[h*width+w].green; 
          data(2,h,w) = pixels[h*width+w].blue; 
        }
    }
    return db::detail::InlinedArrayImpl(data);
  }
  else
  {
    throw db::ImageException();
  }
}

void db::ImageArrayCodec::save (const std::string& filename,
    const db::detail::InlinedArrayImpl& data) const {
  //can only save tree-dimensional data, so throw if that is not the case
  if (data.getNDim() != 3) throw db::DimensionError(data.getNDim(), 3);

  if(data.getElementType() != Torch::core::array::t_uint16)
    throw db::ImageException();

  const size_t *shape = data.getShape();
  if (shape[0] != 1 && shape[0] != 3) throw db::ImageException();

  //
  int height = shape[1];
  int width = shape[2];

  // Write image
  Magick::Image image;
  blitz::Array<uint16_t,3> img = data.get<uint16_t,3>();
  if( shape[0] == 1) // Grayscale
  {
    uint16_t *pixels = new uint16_t[width*height];
    for (int h=0; h<height; ++h)
      for (int w=0; w<width; ++w)
        pixels[h*width+w] = img(0,h,w);
    image.read( shape[2], shape[1], "I", Magick::ShortPixel, pixels );
    image.write( filename.c_str());
    delete [] pixels;
  }
  else if( shape[0] == 3) // RGB
  {
    // Ensure that there is only one reference to underlying image
    // If this is not done, then image pixels will not be modified.
    image.modifyImage();
    // Allocate pixel view
    Magick::Pixels view(image); 
    Magick::PixelPacket *pixels = view.get(0,0,width,height);
    for (int h=0; h<height; ++h )
      for (int w=0; w<width; ++w ) {
        pixels[h*width+w].red = img(0,h,w); 
        pixels[h*width+w].green = img(1,h,w); 
        pixels[h*width+w].blue = img(2,h,w); 
      }
     // Save changes to image.
     view.sync();
  }
}
