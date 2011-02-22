/**
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Implements a JPEG image format reader/writer
 *
 * This codec will only be able to work with three-dimension input.
 */

#include "database/JPEGImageArrayCodec.h"
#include "database/ArrayCodecRegistry.h"
#include "database/Exception.h"
#include <fstream>
#include <unistd.h>
#include <jpeglib.h>
#include <cstdio>

namespace db = Torch::database;

namespace Torch { namespace database {
  class JPEGImageException: public Torch::core::Exception { };
}}

//Takes care of the codec registration.
static bool register_codec() {
  db::ArrayCodecRegistry::addCodec(boost::shared_ptr<db::ArrayCodec>(new db::JPEGImageArrayCodec())); 
  return true;
}

static bool codec_registered = register_codec();

db::JPEGImageArrayCodec::JPEGImageArrayCodec()
  : m_name("torch.image.jpg"),
    m_extensions()
{ 
  m_extensions.push_back(".jpg");
  m_extensions.push_back(".jpeg");
}

db::JPEGImageArrayCodec::~JPEGImageArrayCodec() { }

void db::JPEGImageArrayCodec::peek(const std::string& filename, 
    Torch::core::array::ElementType& eltype, size_t& ndim,
    size_t* shape) const 
{
  // Declare jpeg structures
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  // Initialization for decompression
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);

  // Open the file
  FILE* infile;
  if ( !(infile = fopen(filename.c_str(), "rb")) )
    throw db::FileNotReadable(filename); 

  // Process the header
  jpeg_stdio_src(&cinfo, infile);
  (void) jpeg_read_header(&cinfo, true);
  // Get the with and the height and the number of channels
  shape[0] = cinfo.num_components;
  shape[1] = cinfo.image_height;
  shape[2] = cinfo.image_width;

  // Deallocate the memory
  jpeg_destroy_decompress(&cinfo);

  // Close the file 
  fclose(infile);

  // Set other attributes
  eltype = Torch::core::array::t_uint8;
  ndim = 3;
}

db::detail::InlinedArrayImpl 
db::JPEGImageArrayCodec::load(const std::string& filename) const {
  // Declare jpeg structures
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  // Initialization for decompression
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);

  // Open the file
  FILE* infile;
  if ( !(infile = fopen(filename.c_str(), "rb")) )
    throw db::FileNotReadable(filename); 

  // Process the header and start the decompression
  jpeg_stdio_src(&cinfo, infile);
  (void) jpeg_read_header(&cinfo, true);
  (void) jpeg_start_decompress(&cinfo);
  // Get the with and the height and the number of channels
  int n_components = cinfo.out_color_components;
  int height = cinfo.output_height;
  int width = cinfo.output_width;

  // Create the blitz array
  blitz::Array<uint8_t,3> data(n_components, height, width);

  // Parse the data
  int row_stride = n_components*width;
  JSAMPARRAY pJpegBuffer = (*cinfo.mem->alloc_sarray)
    ((j_common_ptr) &cinfo, JPOOL_IMAGE, row_stride, 1);

  int y = 0;
  while(cinfo.output_scanline < static_cast<size_t>(height)) {
    (void) jpeg_read_scanlines(&cinfo, pJpegBuffer, 1);
    for(int x=0; x<width; ++x) 
      for(int c=0; c<n_components; ++c)
        data(c,y,x) = pJpegBuffer[0][cinfo.output_components*x+c];
    ++y;
  }

  // Deallocate the memory
  (void) jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);

  // Close the file 
  fclose(infile);

  return db::detail::InlinedArrayImpl(data);
}

void db::JPEGImageArrayCodec::save (const std::string& filename,
    const db::detail::InlinedArrayImpl& data) const {
  //can only save two-dimensional data, so throw if that is not the case
  if (data.getNDim() != 3) throw db::DimensionError(data.getNDim(), 3);

  const size_t *shape = data.getShape();
  if (shape[0] != 1 && shape[0] != 3) throw db::JPEGImageException();

  // Declare jpeg structures
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  // Initialization for decompression
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);

  // Open the file
  FILE* outfile;
  if( !(outfile = fopen(filename.c_str(), "wb")) )
    throw db::FileNotReadable(filename); 
  
  // Initialize the encoding
  jpeg_stdio_dest(&cinfo, outfile);

  // Initialize compression parameters
  int quality = 95; 
  jpeg_set_quality(&cinfo, quality, true);
  cinfo.image_width = shape[2]; // image width and height, in pixels
  cinfo.image_height = shape[1];
  cinfo.input_components = shape[0];     // # of color components per pixel
  if(shape[0]==1)
    cinfo.in_color_space = JCS_GRAYSCALE; // colorspace of input image
  else if(shape[0]==3)
    cinfo.in_color_space = JCS_RGB; // colorspace of input image
  else
    throw db::JPEGImageException();
  cinfo.optimize_coding = 100;
  cinfo.smoothing_factor = 0;
  jpeg_set_defaults(&cinfo);

  // Start the compression
  jpeg_start_compress(&cinfo, true);


  // Fill the pixmap with the image pixels
  blitz::Array<uint8_t,3> img = data.get<uint8_t,3>();
  // Get the with and the height and the number of channels
  int n_components = shape[0];
  int height = shape[1];
  int width = shape[2];
  unsigned char* pixmap = new unsigned char[n_components*height*width];
  for(int h=0; h<height; ++h)
    for(int w=0; w<width; ++w)
      for(int c=0; c<n_components; ++c)
        pixmap[h*width*n_components+w*n_components+c] = img(c,h,w);

  // Process the pixmap
  JSAMPROW row_pointer[1];            // pointer to a single row
  int row_stride = shape[0]*shape[2]; // physical row width in buffer
  while (cinfo.next_scanline < cinfo.image_height) {
    row_pointer[0] = & pixmap[cinfo.next_scanline * row_stride];
    jpeg_write_scanlines(&cinfo, row_pointer, 1);
  }
  delete [] pixmap;

  // Deallocate the memory
  (void) jpeg_finish_compress(&cinfo);
  jpeg_destroy_compress(&cinfo);

  // Close the file 
  fclose(outfile);
}
