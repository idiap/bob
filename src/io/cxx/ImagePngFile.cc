/**
 * @file io/cxx/ImagePngFile.cc
 * @date Fri Oct 12 12:08:00 2012 +0200
 * @author Laurent El Shafey <laurent.el-shafey@idiap.ch>
 *
 * @brief Implements an image format reader/writer using libpng.
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

#include <boost/filesystem.hpp>
#include <boost/shared_array.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <string>

#include "bob/io/CodecRegistry.h"
#include "bob/io/Exception.h"

extern "C" {
#include <png.h>
}

// The png_jmpbuf() macro, used in error handling, became available in
// libpng version 1.0.6. In order to be able to run the code with older
// versions of libpng, we define the following macro (but only if it
// is not already defined by libpng!).
#ifndef png_jmpbuf
#  define png_jmpbuf(png_ptr) ((png_ptr)->png_jmpbuf)
#endif


static boost::shared_ptr<std::FILE> make_cfile(const char *filename, const char *flags)
{
  std::FILE* fp = std::fopen(filename, flags);
  if(fp == 0) throw bob::io::FileNotReadable(filename);
  return boost::shared_ptr<std::FILE>(fp, std::fclose);
}

/**
 * LOADING
 */
static void im_peek(const std::string& path, bob::core::array::typeinfo& info) 
{
  // 1. PNG structure declarations
  png_structp png_ptr;
  png_infop info_ptr;

  // 2. PNG file opening
  boost::shared_ptr<std::FILE> in_file = make_cfile(path.c_str(), "rb");

  // 3. Create and initialize the png_struct. The compiler header file version
  //    is supplied, so that we know if the application was compiled with a 
  //    compatible version of the library.
  png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if(png_ptr == NULL) throw std::runtime_error("PNG: error while creating read png structure (function png_create_read_struct())");

  // Allocate/initialize the memory for image information.
  info_ptr = png_create_info_struct(png_ptr);
  if(info_ptr == NULL) { 
    png_destroy_read_struct(&png_ptr, NULL, NULL); 
    throw std::runtime_error("PNG: error while creating info png structure (function png_create_info_struct())");
  }

  // 4. Set error handling if you are using the setjmp/longjmp method (this is
  // the normal method of doing things with libpng). This is required as we
  // did not set up our own error handlers in the png_create_read_struct() earlier.
  if(setjmp(png_jmpbuf(png_ptr)))
  {
    // Free all of the memory associated with the png_ptr and info_ptr
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    throw std::runtime_error("PNG: error while setting error handling (function setjmp(png_jmpbuf(()))");
  }

  // 5. Initialize
  png_init_io(png_ptr, in_file.get());

  // 7. The call to png_read_info() gives us all of the information from the
  // PNG file.
  png_read_info(png_ptr, info_ptr);
  // Get header information
  png_uint_32 width, height;
  int bit_depth, color_type, interlace_type;
  png_get_IHDR(png_ptr, info_ptr, &width, &height, &bit_depth, &color_type,
    &interlace_type, NULL, NULL);

  // 8. Clean up after the read, and free any memory allocated
  png_destroy_read_struct(&png_ptr, &info_ptr, NULL);

  // Set depth and number of dimensions
  info.dtype = (bit_depth <= 8 ? bob::core::array::t_uint8 : bob::core::array::t_uint16);
  if(color_type == PNG_COLOR_TYPE_GRAY)
    info.nd = 2;
  else if (color_type == PNG_COLOR_TYPE_RGB)
    info.nd = 3;
  else // Unsupported color type
    throw bob::io::ImageUnsupportedColorspace();
  if(info.nd == 2)
  {
    info.shape[0] = height;
    info.shape[1] = width;
  }
  else
  {
    info.shape[0] = 3;
    info.shape[1] = height;
    info.shape[2] = width;
  }
  info.update_strides();
}

template <typename T> static
void im_load_gray(png_structp png_ptr, bob::core::array::interface& b) 
{
  const bob::core::array::typeinfo& info = b.type();
  const size_t height = info.shape[0];
  const size_t width = info.shape[1];

#ifdef PNG_READ_INTERLACING_SUPPORTED
  // Turn on interlace handling.  
  int number_passes = png_set_interlace_handling(png_ptr);
#else
  int number_passes = 1;
#endif // PNG_READ_INTERLACING_SUPPORTED 

  // Read the image (one row at a time)
  // This can deal with interlacing
  png_bytep ptr;
  for(int pass=0; pass<number_passes; ++pass)
  {
    // Loop over the rows
    for(size_t y=0; y<height; ++y) 
    {
      ptr = reinterpret_cast<png_bytep>(reinterpret_cast<T*>(b.ptr())+y*width);
      png_read_row(png_ptr, ptr, NULL);
    }
  }
}

template <typename T> static
void imbuffer_to_rgb(const size_t size, const T* im, T* r, T* g, T* b) 
{
  for(size_t k=0; k<size; ++k) 
  {
    r[k] = im[3*k];
    g[k] = im[3*k +1];
    b[k] = im[3*k +2];
  }
}

template <typename T> static
void im_load_color(png_structp png_ptr, bob::core::array::interface& b) 
{
  const bob::core::array::typeinfo& info = b.type();
  const size_t height = info.shape[1];
  const size_t width = info.shape[2];
  const size_t frame_size = height * width;
  const size_t row_color_stride = width;

  // Allocate array to contains a row of RGB-like pixels
  boost::shared_array<T> row(new T[3*width]);
  png_bytep row_pointer = reinterpret_cast<png_bytep>(row.get()); 

#ifdef PNG_READ_INTERLACING_SUPPORTED
  // Turn on interlace handling.  
  int number_passes = png_set_interlace_handling(png_ptr);
#else
  int number_passes = 1;
#endif // PNG_READ_INTERLACING_SUPPORTED 

  // Read the image (one row at a time)
  // This can deal with interlacing
  T *element_r;
  T *element_g;
  T *element_b;
  for(int pass=0; pass<number_passes; ++pass)
  {
    element_r = reinterpret_cast<T*>(b.ptr());
    element_g = element_r + frame_size;
    element_b = element_g + frame_size;
    // Loop over the rows
    for(size_t y=0; y<height; ++y)
    {
      png_read_row(png_ptr, row_pointer, NULL);
      imbuffer_to_rgb(row_color_stride, reinterpret_cast<T*>(row_pointer), element_r, element_g, element_b);
      element_r += row_color_stride;
      element_g += row_color_stride;
      element_b += row_color_stride;
    }
  }
}

static void im_load(const std::string& filename, bob::core::array::interface& b) 
{
  // 1. PNG structure declarations
  png_structp png_ptr;
  png_infop info_ptr;

  // 2. PNG file opening
  boost::shared_ptr<std::FILE> in_file = make_cfile(filename.c_str(), "rb");

  // 3. Create and initialize the png_struct with the desired error handler
  // functions. The compiler header file version is supplied, so that we 
  // know if the application was compiled with a compatible version of the library.
  png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if(png_ptr == NULL) throw std::runtime_error("PNG: error while creating read png structure (function png_create_read_struct())");

  // Allocate/initialize the memory for image informatio
  info_ptr = png_create_info_struct(png_ptr);
  if(info_ptr == NULL) { 
    png_destroy_read_struct(&png_ptr, NULL, NULL); 
    throw std::runtime_error("PNG: error while creating info png structure (function png_create_info_struct())");
  }

  // 4. Set error handling if you are using the setjmp/longjmp method (this is
  // the normal method of doing things with libpng). This is required as we did
  // not set up your own error handlers in the png_create_read_struct() earlier.
  if(setjmp(png_jmpbuf(png_ptr)))
  {
    // Free all of the memory associated with the png_ptr and info_ptr
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    throw std::runtime_error("PNG: error while setting error handling (function setjmp(png_jmpbuf(()))");
  }

  // 5. Initialize
  png_init_io(png_ptr, in_file.get());

  // 6. The call to png_read_info() gives us all of the information from the
  // PNG file.
  png_read_info(png_ptr, info_ptr);
  // Get header information
  png_uint_32 width, height;
  int bit_depth, color_type, interlace_type;
  png_get_IHDR(png_ptr, info_ptr, &width, &height, &bit_depth, &color_type,
       &interlace_type, NULL, NULL);

  // Extract multiple pixels with bit depths of 1, 2, and 4 from a single
  // byte into separate bytes (useful for paletted and grayscale images).
  png_set_packing(png_ptr);

  // Expand grayscale images to the full 8 bits from 1, 2, or 4 bits/pixel
  if(color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
    png_set_expand_gray_1_2_4_to_8(png_ptr);

  // We currently only support grayscale and rgb images
  if(color_type != PNG_COLOR_TYPE_GRAY && color_type != PNG_COLOR_TYPE_RGB)
    throw bob::io::ImageUnsupportedColorspace();

  // 7. Read content
  const bob::core::array::typeinfo& info = b.type();
  if(info.dtype == bob::core::array::t_uint8) {
    if(info.nd == 2) im_load_gray<uint8_t>(png_ptr, b);
    else if( info.nd == 3) im_load_color<uint8_t>(png_ptr, b); 
    else { 
      png_destroy_read_struct(&png_ptr, &info_ptr, NULL); 
      throw bob::io::ImageUnsupportedDimension(info.nd);
    }
  }
  else if(info.dtype == bob::core::array::t_uint16) {
    if(info.nd == 2) im_load_gray<uint16_t>(png_ptr, b);
    else if( info.nd == 3) im_load_color<uint16_t>(png_ptr, b); 
    else { 
      png_destroy_read_struct(&png_ptr, &info_ptr, NULL); 
      throw bob::io::ImageUnsupportedDimension(info.nd);
    }
  }
  else {
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL); 
    throw bob::io::ImageUnsupportedType();
  }

  // 8. Clean up after the read, and free any memory allocated
  // Read rest of file, and get additional chunks in info_ptr
  png_read_end(png_ptr, NULL);
  png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
}


/**
 * SAVING
 */
template <typename T>
static void im_save_gray(const bob::core::array::interface& b, png_structp png_ptr) 
{
  const bob::core::array::typeinfo& info = b.type();
  const size_t height = info.shape[0];
  const size_t width = info.shape[1];  

  const T* row_pointer = reinterpret_cast<const T*>(b.ptr());
  png_bytep row_pointer_;

  // Save one row at a time
  for(size_t y=0; y<height; ++y) 
  {
    row_pointer_ = reinterpret_cast<png_bytep>(const_cast<T*>(row_pointer));
    png_write_row(png_ptr, row_pointer_);
    row_pointer += width;
  }
}

template <typename T> static
void rgb_to_imbuffer(const size_t size, const T* r, const T* g, const T* b, T* im) 
{
  for (size_t k=0; k<size; ++k) 
  {
    im[3*k]   = r[k];
    im[3*k+1] = g[k];
    im[3*k+2] = b[k];
  }
}

template <typename T>
static void im_save_color(const bob::core::array::interface& b, png_structp png_ptr) 
{
  const bob::core::array::typeinfo& info = b.type();
  const size_t height = info.shape[1];
  const size_t width = info.shape[2];
  const size_t frame_size = height * width;
  const size_t row_color_stride = width;

  // Allocate array for a row as an RGB-like array
  boost::shared_array<T> row(new T[3*width]);
  png_bytep array_ptr = reinterpret_cast<png_bytep>(row.get()); 

  // pointer to a single row (png_bytep is a typedef to unsigned char or char)
  const T *element_r = static_cast<const T*>(b.ptr());
  const T *element_g = element_r + frame_size;
  const T *element_b = element_g + frame_size;
  for(size_t y=0; y<height; ++y) 
  {
    rgb_to_imbuffer(row_color_stride, element_r, element_g, element_b, reinterpret_cast<T*>(array_ptr));
    png_write_row(png_ptr, array_ptr);
    element_r += row_color_stride;
    element_g += row_color_stride;
    element_b += row_color_stride;
  }
}

static void im_save(const std::string& filename, const bob::core::array::interface& array) 
{
  // 1. PNG structures
  png_structp png_ptr;
  png_infop info_ptr;

  // 2. Open the file
  boost::shared_ptr<std::FILE> out_file = make_cfile(filename.c_str(), "wb");

  // 3.Create and initialize the png_struct with the desired error handler functions.
  png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

  // Allocate/initialize the image information data.
  info_ptr = png_create_info_struct(png_ptr);
  if(info_ptr == NULL)
  {
    png_destroy_write_struct(&png_ptr,  NULL);
    throw std::runtime_error("PNG: error while creating info png structure (function png_create_info_struct())");
  }

  // 4. Set error handling. 
  // REQUIRED when an error handling function is not passed in in the 
  //  png_create_write_struct() call.
  if(setjmp(png_jmpbuf(png_ptr)))
  {
    // If we get here, we had a problem writing the file
    png_destroy_write_struct(&png_ptr, &info_ptr);
    throw std::runtime_error("PNG: error while setting error handling (function setjmp(png_jmpbuf(()))");
  }

  // 5. Initialize
  png_init_io(png_ptr, out_file.get());

  // 6. Set the image information here:
  // width and height are up to 2^31
  // bit_depth is one of 1, 2, 4, 8, or 16, but valid values also depend on the color_type selected
  // color_type is one of PNG_COLOR_TYPE_GRAY, PNG_COLOR_TYPE_GRAY_ALPHA, PNG_COLOR_TYPE_PALETTE, PNG_COLOR_TYPE_RGB,
  // or PNG_COLOR_TYPE_RGB_ALPHA
  // interlace is either PNG_INTERLACE_NONE or PNG_INTERLACE_ADAM7
  // compression_type and filter_type MUST currently be PNG_COMPRESSION_TYPE_DEFAULT and PNG_FILTER_TYPE_DEFAULT
  const bob::core::array::typeinfo& info = array.type();
  png_uint_32 height = (info.nd == 2 ? info.shape[0] : info.shape[1]);
  png_uint_32 width = (info.nd == 2 ? info.shape[1] : info.shape[2]);
  int bit_depth = (info.dtype == bob::core::array::t_uint8 ? 8 : 16);
  png_set_IHDR(png_ptr, info_ptr, width, height, bit_depth, 
    (info.nd == 2 ? PNG_COLOR_TYPE_GRAY : PNG_COLOR_TYPE_RGB),
    PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

  // Write the file header information. 
  png_write_info(png_ptr, info_ptr);

  // Pack pixels into bytes
  png_set_packing(png_ptr);

  // 7. Writes content
  if(info.dtype == bob::core::array::t_uint8) {
    if(info.nd == 2) im_save_gray<uint8_t>(array, png_ptr);
    else if(info.nd == 3) {
      if(info.shape[0] != 3) 
      {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        throw std::runtime_error("color image does not have 3 planes on 1st. dimension");
      }
      im_save_color<uint8_t>(array, png_ptr);
    }
    else 
    {
      png_destroy_write_struct(&png_ptr, &info_ptr);
      throw bob::io::ImageUnsupportedDimension(info.nd); 
    }
  }
  else if(info.dtype == bob::core::array::t_uint16) {
    if(info.nd == 2) im_save_gray<uint16_t>(array, png_ptr);
    else if(info.nd == 3) {
      if(info.shape[0] != 3) 
      {
      png_destroy_write_struct(&png_ptr, &info_ptr);
        throw std::runtime_error("color image does not have 3 planes on 1st. dimension");
      }
      im_save_color<uint16_t>(array, png_ptr);
    }
    else
    {
      png_destroy_write_struct(&png_ptr, &info_ptr);
      throw bob::io::ImageUnsupportedDimension(info.nd);
    }
  }
  else 
  {
    png_destroy_write_struct(&png_ptr, &info_ptr);
    throw bob::io::ImageUnsupportedType();
  }

  // It is REQUIRED to call this to finish writing the rest of the file
  png_write_end(png_ptr, NULL);

  // Clean up after the write, and free any memory allocated
  png_destroy_write_struct(&png_ptr, &info_ptr);
}

class ImagePngFile: public bob::io::File {

  public: //api

    ImagePngFile(const std::string& path, char mode):
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

    virtual ~ImagePngFile() { }

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

std::string ImagePngFile::s_codecname = "bob.image_png";

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
  return boost::make_shared<ImagePngFile>(path, mode);
}

/**
 * Takes care of codec registration per se.
 */
static bool register_codec() {
  boost::shared_ptr<bob::io::CodecRegistry> instance =
    bob::io::CodecRegistry::instance();

  instance->registerExtension(".png", "PNG, compressed (libpng)", &make_file);

  return true;
}

static bool codec_registered = register_codec();

