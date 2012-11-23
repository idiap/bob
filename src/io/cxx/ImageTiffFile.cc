/**
 * @file io/cxx/ImageTiffFile.cc
 * @date Fri Oct 12 12:08:00 2012 +0200
 * @author Laurent El Shafey <laurent.el-shafey@idiap.ch>
 *
 * @brief Implements an image format reader/writer using libtiff.
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
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <string>

#include "bob/io/CodecRegistry.h"
#include "bob/io/Exception.h"

extern "C" {
#include <tiffio.h>
}

static boost::shared_ptr<TIFF> make_cfile(const char *filename, const char *flags)
{
  TIFF* fp = TIFFOpen(filename, flags);
  if(fp == 0) throw bob::io::FileNotReadable(filename);
  return boost::shared_ptr<TIFF>(fp, TIFFClose);
}

/**
 * LOADING
 */
static void im_peek(const std::string& path, bob::core::array::typeinfo& info) 
{
  // 1. TIFF file opening
  boost::shared_ptr<TIFF> in_file = make_cfile(path.c_str(), "r");

  // 2. Get file information
  uint32 w, h;
  TIFFGetField(in_file.get(), TIFFTAG_IMAGEWIDTH, &w);
  TIFFGetField(in_file.get(), TIFFTAG_IMAGELENGTH, &h);
  size_t width = (size_t)w;
  size_t height = (size_t)h;

  uint16 bps, spp;
  TIFFGetField(in_file.get(), TIFFTAG_BITSPERSAMPLE, &bps);
  TIFFGetField(in_file.get(), TIFFTAG_SAMPLESPERPIXEL, &spp);

  // 3. Set typeinfo variables
  info.dtype = (bps <= 8 ? bob::core::array::t_uint8 : bob::core::array::t_uint16);
  if(spp == 1)
    info.nd = 2;
  else if (spp == 3)
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
void im_load_gray(boost::shared_ptr<TIFF> in_file, bob::core::array::interface& b) 
{
  const bob::core::array::typeinfo& info = b.type();
  const size_t height = info.shape[0];
  const size_t width = info.shape[1];

  // Read in the possibly multiple strips
  tsize_t strip_size = TIFFStripSize(in_file.get());
  tstrip_t n_strips = TIFFNumberOfStrips(in_file.get());
  
  unsigned long buffer_size = n_strips * strip_size;
  boost::shared_array<unsigned char> buffer_(new unsigned char[buffer_size]);
  unsigned char* buffer = buffer_.get();
  if(buffer == 0) throw bob::io::Exception();
  
  tsize_t result;
  tsize_t image_offset = 0;
  for(tstrip_t strip_count=0; strip_count<n_strips; ++strip_count) 
  {
    if((result = TIFFReadEncodedStrip(in_file.get(), strip_count, buffer+image_offset, strip_size)) == -1)
      throw bob::io::Exception();
    image_offset += result;
  }

  // Deal with photometric interpretations
  uint16 photo = PHOTOMETRIC_MINISBLACK;
  if(TIFFGetField(in_file.get(), TIFFTAG_PHOTOMETRIC, &photo) == 0 || (photo != PHOTOMETRIC_MINISBLACK && photo != PHOTOMETRIC_MINISWHITE))
    throw bob::io::Exception(); 

  if(photo != PHOTOMETRIC_MINISBLACK) 
  {
    // Flip bits
    for(unsigned long count=0; count<buffer_size; ++count)
      buffer[count] = ~buffer[count];
  }

  // Deal with fillorder
  uint16 fillorder = FILLORDER_MSB2LSB;
  TIFFGetField(in_file.get(), TIFFTAG_FILLORDER, &fillorder);
  
  if(fillorder != FILLORDER_MSB2LSB) {
    // We need to swap bits -- ABCDEFGH becomes HGFEDCBA
    for(unsigned long count=0; count<buffer_size; ++count)
    {
      unsigned char tempbyte = 0;
      if(buffer[count] & 128) tempbyte += 1;
      if(buffer[count] & 64) tempbyte += 2;
      if(buffer[count] & 32) tempbyte += 4;
      if(buffer[count] & 16) tempbyte += 8;
      if(buffer[count] & 8) tempbyte += 16;
      if(buffer[count] & 4) tempbyte += 32;
      if(buffer[count] & 2) tempbyte += 64;
      if(buffer[count] & 1) tempbyte += 128;
      buffer[count] = tempbyte;
    }
  }

  // Copy to output array
  T *element = reinterpret_cast<T*>(b.ptr());
  T *b_in = reinterpret_cast<T*>(buffer);
  memcpy(element, b_in, height*width*sizeof(T));
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
void im_load_color(boost::shared_ptr<TIFF> in_file, bob::core::array::interface& b) 
{
  const bob::core::array::typeinfo& info = b.type();
  const size_t height = info.shape[1];
  const size_t width = info.shape[2];
  const size_t frame_size = height*width;
  const size_t row_stride = width;
  const size_t row_color_stride = 3*width;

  // Read in the possibly multiple strips
  tsize_t strip_size = TIFFStripSize(in_file.get());
  tstrip_t n_strips = TIFFNumberOfStrips(in_file.get());
  
  unsigned long buffer_size = n_strips * strip_size;
  boost::shared_array<unsigned char> buffer_(new unsigned char[buffer_size]);
  unsigned char* buffer = buffer_.get();
  if(buffer == 0) throw bob::io::Exception();
  
  tsize_t result;
  tsize_t image_offset = 0;
  for(tstrip_t strip_count=0; strip_count<n_strips; ++strip_count) 
  {
    if((result = TIFFReadEncodedStrip(in_file.get(), strip_count, buffer+image_offset, strip_size)) == -1)
      throw bob::io::Exception();

    image_offset += result;
  }

  // Deal with photometric interpretations
  uint16 photo = PHOTOMETRIC_RGB;
  if(TIFFGetField(in_file.get(), TIFFTAG_PHOTOMETRIC, &photo) == 0 || photo != PHOTOMETRIC_RGB)
    throw bob::io::Exception(); 

  // Deal with fillorder
  uint16 fillorder = FILLORDER_MSB2LSB;
  TIFFGetField(in_file.get(), TIFFTAG_FILLORDER, &fillorder);
  
  if(fillorder != FILLORDER_MSB2LSB) {
    // We need to swap bits -- ABCDEFGH becomes HGFEDCBA
    for(unsigned long count=0; count<(unsigned long)image_offset; ++count)
    {
      unsigned char tempbyte = 0;
      if(buffer[count] & 128) tempbyte += 1;
      if(buffer[count] & 64) tempbyte += 2;
      if(buffer[count] & 32) tempbyte += 4;
      if(buffer[count] & 16) tempbyte += 8;
      if(buffer[count] & 8) tempbyte += 16;
      if(buffer[count] & 4) tempbyte += 32;
      if(buffer[count] & 2) tempbyte += 64;
      if(buffer[count] & 1) tempbyte += 128;
      buffer[count] = tempbyte;
    }
  }

  // Read the image (one row at a time)
  // This can deal with interlacing
  T *element_r = reinterpret_cast<T*>(b.ptr());
  T *element_g = element_r + frame_size;
  T *element_b = element_g + frame_size;
  unsigned char *row_pointer = buffer;
  // Loop over the rows
  for(size_t y=0; y<height; ++y)
  {
    imbuffer_to_rgb(row_stride, reinterpret_cast<T*>(row_pointer), element_r, element_g, element_b);
    element_r += row_stride;
    element_g += row_stride;
    element_b += row_stride;
    row_pointer += row_color_stride * sizeof(T);
  }
}

static void im_load(const std::string& filename, bob::core::array::interface& b) 
{
  // 1. TIFF file opening
  boost::shared_ptr<TIFF> in_file = make_cfile(filename.c_str(), "r");

  // 2. Read content
  const bob::core::array::typeinfo& info = b.type();
  if(info.dtype == bob::core::array::t_uint8) {
    if(info.nd == 2) im_load_gray<uint8_t>(in_file, b);
    else if( info.nd == 3) im_load_color<uint8_t>(in_file, b); 
    else { 
      throw bob::io::ImageUnsupportedDimension(info.nd);
    }
  }
  else if(info.dtype == bob::core::array::t_uint16) {
    if(info.nd == 2) im_load_gray<uint16_t>(in_file, b);
    else if( info.nd == 3) im_load_color<uint16_t>(in_file, b); 
    else { 
      throw bob::io::ImageUnsupportedDimension(info.nd);
    }
  }
  else {
    throw bob::io::ImageUnsupportedType(info.dtype);
  }
}


/**
 * SAVING
 */
template <typename T>
static void im_save_gray(const bob::core::array::interface& b, boost::shared_ptr<TIFF> out_file) 
{
  const bob::core::array::typeinfo& info = b.type();
  const size_t height = info.shape[0];
  const size_t width = info.shape[1];  

  unsigned char* row_pointer = const_cast<unsigned char*>(reinterpret_cast<const unsigned char*>(b.ptr()));
  const size_t data_size = height * width * sizeof(T);

  // Write the information to the file
  TIFFWriteEncodedStrip(out_file.get(), 0, row_pointer, data_size);
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
static void im_save_color(const bob::core::array::interface& b, boost::shared_ptr<TIFF> out_file)
{
  const bob::core::array::typeinfo& info = b.type();
  const size_t height = info.shape[1];
  const size_t width = info.shape[2];
  const size_t frame_size = height * width;

  // Allocate array for a row as an RGB-like array
  boost::shared_array<T> row(new T[3*width*height]);
  unsigned char* row_pointer = reinterpret_cast<unsigned char*>(row.get());

  // pointer to a single row (tiff_bytep is a typedef to unsigned char or char)
  const T *element_r = static_cast<const T*>(b.ptr());
  const T *element_g = element_r + frame_size;
  const T *element_b = element_g + frame_size;
  rgb_to_imbuffer(frame_size, element_r, element_g, element_b, row.get());

  // Write the information to the file
  const size_t data_size = 3 * height * width * sizeof(T);
  TIFFWriteEncodedStrip(out_file.get(), 0, row_pointer, data_size);
}

static void im_save(const std::string& filename, const bob::core::array::interface& array) 
{
  // 1. Open the file
  boost::shared_ptr<TIFF> out_file = make_cfile(filename.c_str(), "w");

  // 2. Set the image information here:
  const bob::core::array::typeinfo& info = array.type();
  const int height = (info.nd == 2 ? info.shape[0] : info.shape[1]);
  const int width = (info.nd == 2 ? info.shape[1] : info.shape[2]);
  TIFFSetField(out_file.get(), TIFFTAG_IMAGELENGTH, height);
  TIFFSetField(out_file.get(), TIFFTAG_IMAGEWIDTH, width);
  TIFFSetField(out_file.get(), TIFFTAG_BITSPERSAMPLE, (info.dtype == bob::core::array::t_uint8 ? 8 : 16));
  TIFFSetField(out_file.get(), TIFFTAG_SAMPLESPERPIXEL, (info.nd == 2 ? 1 : 3));

  TIFFSetField(out_file.get(), TIFFTAG_COMPRESSION, COMPRESSION_NONE);
  TIFFSetField(out_file.get(), TIFFTAG_FILLORDER, FILLORDER_MSB2LSB);
  if(info.nd == 3)
    TIFFSetField(out_file.get(), TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
  TIFFSetField(out_file.get(), TIFFTAG_PHOTOMETRIC, (info.nd == 2 ? PHOTOMETRIC_MINISBLACK : PHOTOMETRIC_RGB));

  // 3. Writes content
  if(info.dtype == bob::core::array::t_uint8) {
    if(info.nd == 2) im_save_gray<uint8_t>(array, out_file);
    else if(info.nd == 3) {
      if(info.shape[0] != 3) 
        throw std::runtime_error("color image does not have 3 planes on 1st. dimension");
      im_save_color<uint8_t>(array, out_file);
    }
    else 
      throw bob::io::ImageUnsupportedDimension(info.nd); 
  }
  else if(info.dtype == bob::core::array::t_uint16) {
    if(info.nd == 2) im_save_gray<uint16_t>(array, out_file);
    else if(info.nd == 3) {
      if(info.shape[0] != 3) 
        throw std::runtime_error("color image does not have 3 planes on 1st. dimension");
      im_save_color<uint16_t>(array, out_file);
    }
    else
      throw bob::io::ImageUnsupportedDimension(info.nd);
  }
  else 
    throw bob::io::ImageUnsupportedType(info.dtype);
}

class ImageTiffFile: public bob::io::File {

  public: //api

    ImageTiffFile(const std::string& path, char mode):
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

    virtual ~ImageTiffFile() { }

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

std::string ImageTiffFile::s_codecname = "bob.image_tiff";

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
  return boost::make_shared<ImageTiffFile>(path, mode);
}

/**
 * Takes care of codec registration per se.
 */
static bool register_codec() {
  boost::shared_ptr<bob::io::CodecRegistry> instance =
    bob::io::CodecRegistry::instance();

  instance->registerExtension(".tif", "TIFF, compressed (libtiff)", &make_file);
  instance->registerExtension(".tiff", "TIFF, compressed (libtiff)", &make_file);

  return true;
}

static bool codec_registered = register_codec();

