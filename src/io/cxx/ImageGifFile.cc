/**
 * @file io/cxx/ImageGifFile.cc
 * @date Fri Nov 23 16:53:00 2012 +0200
 * @author Laurent El Shafey <laurent.el-shafey@idiap.ch>
 *
 * @brief Implements an image format reader/writer using giflib.
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
#include <gif_lib.h>
}


static boost::shared_ptr<GifFileType> make_dfile(const char *filename)
{
  int error;
  GifFileType* fp = DGifOpenFileName(filename, &error);
  if(fp == 0) throw bob::io::FileNotReadable(filename);
  return boost::shared_ptr<GifFileType>(fp, DGifCloseFile);
}

static boost::shared_ptr<GifFileType> make_efile(const char *filename)
{
  int error;
  GifFileType* fp = EGifOpenFileName(filename, false, &error);
  if(fp == 0) throw bob::io::FileNotReadable(filename);
  return boost::shared_ptr<GifFileType>(fp, EGifCloseFile);
}

/**
 * LOADING
 */
static void im_peek(const std::string& path, bob::core::array::typeinfo& info) 
{
  // 1. GIF file opening
  boost::shared_ptr<GifFileType> in_file = make_dfile(path.c_str());

  // 2. Set typeinfo variables
  info.dtype = bob::core::array::t_uint8;
  info.nd = 3;
  info.shape[0] = 3;
  info.shape[1] = in_file->SHeight;
  info.shape[2] = in_file->SWidth;
  info.update_strides();
}

static void im_load_color(boost::shared_ptr<GifFileType> in_file, bob::core::array::interface& b) 
{ 
  const bob::core::array::typeinfo& info = b.type();
  const size_t height0 = info.shape[1];
  const size_t width0 = info.shape[2];
  const size_t frame_size = height0*width0;
 
  // The following piece of code is based on the giflib utility called gif2rgb
  // Allocate the screen as vector of column of rows. Note this
  // screen is device independent - it's the screen defined by the
  // GIF file parameters. 
  std::vector<boost::shared_array<GifPixelType> > screen_buffer;

  // Size in bytes one row.
  int size = in_file->SWidth*sizeof(GifPixelType);
  // First row
  screen_buffer.push_back(boost::shared_array<GifPixelType>(new GifPixelType[in_file->SWidth]));

  // Set its color to BackGround
  for(int i=0; i<in_file->SWidth; ++i)
    screen_buffer[0][i] = in_file->SBackGroundColor;
  for(int i=1; i<in_file->SHeight; ++i) {
    // Allocate the other rows, and set their color to background too: 
    screen_buffer.push_back(boost::shared_array<GifPixelType>(new GifPixelType[in_file->SWidth]));
    memcpy(screen_buffer[i].get(), screen_buffer[0].get(), size);
  }

  // Scan the content of the GIF file and load the image(s) in: 
  GifRecordType record_type;
  GifByteType *extension;
  int InterlacedOffset[] = { 0, 4, 2, 1 }; // The way Interlaced image should.
  int InterlacedJumps[] = { 8, 8, 4, 2 }; // be read - offsets and jumps...
  int row, col, width, height, count, ext_code;
  if(DGifGetRecordType(in_file.get(), &record_type) == GIF_ERROR) 
    throw std::runtime_error("GIF: error in DGifGetRecordType().");
  switch(record_type) {
    case IMAGE_DESC_RECORD_TYPE:
      if(DGifGetImageDesc(in_file.get()) == GIF_ERROR) 
        throw std::runtime_error("GIF: error in DGifGetImageDesc().");

      row = in_file->Image.Top; // Image Position relative to Screen.
      col = in_file->Image.Left;
      width = in_file->Image.Width;
      height = in_file->Image.Height;
      if(in_file->Image.Left + in_file->Image.Width > in_file->SWidth ||
        in_file->Image.Top + in_file->Image.Height > in_file->SHeight) 
      {
        throw std::runtime_error("GIF: the dimensions of image larger than the dimensions of the canvas.");
      }
      if(in_file->Image.Interlace) {
        // Need to perform 4 passes on the images: 
        for(int i=count=0; i<4; ++i)
          for(int j=row+InterlacedOffset[i]; j<row+height; j+=InterlacedJumps[i]) {
            ++count;
            if(DGifGetLine(in_file.get(), &screen_buffer[j][col], width) == GIF_ERROR) {
              throw std::runtime_error("GIF: error in DGifGetLine().");
            }
          }
      }
      else {
        for(int i=0; i<height; ++i) {
          if(DGifGetLine(in_file.get(), &screen_buffer[row++][col], width) == GIF_ERROR) {
            throw std::runtime_error("GIF: error in DGifGetLine().");
          }
        }
      }
      break;
    case EXTENSION_RECORD_TYPE:
      // Skip any extension blocks in file: 
      if(DGifGetExtension(in_file.get(), &ext_code, &extension) == GIF_ERROR) {
        throw std::runtime_error("GIF: error in DGifGetExtension().");
      }
      while(extension != NULL) {
        if(DGifGetExtensionNext(in_file.get(), &extension) == GIF_ERROR) {
          throw std::runtime_error("GIF: error in DGifGetExtensionNext().");
        }
      }
      break;
    case TERMINATE_RECORD_TYPE:
      break;
    default: // Should be trapped by DGifGetRecordType.
      break;
  }

  // Lets dump it - set the global variables required and do it:
  ColorMapObject *ColorMap = (in_file->Image.ColorMap ? in_file->Image.ColorMap : in_file->SColorMap);
  if(ColorMap == 0)
    throw std::runtime_error("GIF: image does not have a colormap");

  // Put data into C-style buffer
  uint8_t *element_r = reinterpret_cast<uint8_t*>(b.ptr());
  uint8_t *element_g = element_r + frame_size;
  uint8_t *element_b = element_g + frame_size;
  GifRowType gif_row;
  GifColorType *ColorMapEntry;
  for(int i=0; i<in_file->SHeight; ++i) {
    gif_row = screen_buffer[i].get();
    for(int j=0; j<in_file->SWidth; ++j) {
      ColorMapEntry = &ColorMap->Colors[gif_row[j]];
      *element_r++ = ColorMapEntry->Red;
      *element_g++ = ColorMapEntry->Green;
      *element_b++ = ColorMapEntry->Blue;
    }
  }
}

static void im_load(const std::string& filename, bob::core::array::interface& b) 
{
  // 1. GIF file opening
  boost::shared_ptr<GifFileType> in_file = make_dfile(filename.c_str());

  // 2. Read content
  const bob::core::array::typeinfo& info = b.type();
  if(info.dtype == bob::core::array::t_uint8) {
    if( info.nd == 3) im_load_color(in_file, b); 
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
static void im_save_color(const bob::core::array::interface& b, boost::shared_ptr<GifFileType> out_file)
{
  const bob::core::array::typeinfo& info = b.type();
  const int height = info.shape[1];
  const int width = info.shape[2];
  const size_t frame_size = height * width;

  // pointer to a single row (tiff_bytep is a typedef to unsigned char or char)
  const uint8_t *element_r = static_cast<const uint8_t*>(b.ptr());
  const uint8_t *element_g = element_r + frame_size;
  const uint8_t *element_b = element_g + frame_size;

  GifByteType *red_buffer = const_cast<GifByteType*>(reinterpret_cast<const GifByteType*>(element_r));
  GifByteType *green_buffer = const_cast<GifByteType*>(reinterpret_cast<const GifByteType*>(element_g));
  GifByteType *blue_buffer = const_cast<GifByteType*>(reinterpret_cast<const GifByteType*>(element_b));
  boost::shared_array<GifByteType> output_buffer(new GifByteType[width*height]);

  // The following piece of code is based on the giflib utility called gif2rgb
  const int ExpNumOfColors = 8;
  int ColorMapSize = 1 << ExpNumOfColors;
  ColorMapObject *OutputColorMap = 0;

  if((OutputColorMap = GifMakeMapObject(ColorMapSize, NULL)) == 0)
    throw std::runtime_error("GIF: error in GifMakeMapObject().");

  if(GifQuantizeBuffer(width, height, &ColorMapSize,
       red_buffer, green_buffer, blue_buffer, output_buffer.get(), OutputColorMap->Colors) == GIF_ERROR)
    throw std::runtime_error("GIF: error in GifQuantizeBuffer().");

  if(EGifPutScreenDesc(out_file.get(), width, height, ExpNumOfColors, 0, OutputColorMap) == GIF_ERROR)
    throw std::runtime_error("GIF: error in EGifPutScreenDesc().");
 
  if(EGifPutImageDesc(out_file.get(), 0, 0, width, height, false, NULL) == GIF_ERROR)
    throw std::runtime_error("GIF: error in EGifPutImageDesc().");

  GifByteType *ptr = output_buffer.get();
  for(int i=0; i<height; ++i) {
    if(EGifPutLine(out_file.get(), ptr, width) == GIF_ERROR)
      throw std::runtime_error("GIF: error in EGifPutLine().");
    ptr += width;
  }

  // Free map object
  GifFreeMapObject(OutputColorMap);
}

static void im_save(const std::string& filename, const bob::core::array::interface& array) 
{
  // 1. GIF file opening
  boost::shared_ptr<GifFileType> out_file = make_efile(filename.c_str());

  // 2. Set the image information here:
  const bob::core::array::typeinfo& info = array.type();

  // 3. Writes content
  if(info.dtype == bob::core::array::t_uint8) {
    if(info.nd == 3) {
      if(info.shape[0] != 3) 
        throw std::runtime_error("color image does not have 3 planes on 1st. dimension");
      im_save_color(array, out_file);
    }
    else 
      throw bob::io::ImageUnsupportedDimension(info.nd); 
  }
  else 
    throw bob::io::ImageUnsupportedType(info.dtype);
}


class ImageGifFile: public bob::io::File {

  public: //api

    ImageGifFile(const std::string& path, char mode):
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

    virtual ~ImageGifFile() { }

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


std::string ImageGifFile::s_codecname = "bob.image_gif";

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
  return boost::make_shared<ImageGifFile>(path, mode);
}

/**
 * Takes care of codec registration per se.
 */
static bool register_codec() {
  boost::shared_ptr<bob::io::CodecRegistry> instance =
    bob::io::CodecRegistry::instance();

  instance->registerExtension(".gif", "GIF (giflib)", &make_file);

  return true;
}

static bool codec_registered = register_codec();

