/**
 * @file io/cxx/ImageGifFile.cc
 * @date Fri Nov 23 16:53:00 2012 +0200
 * @author Laurent El Shafey <laurent.el-shafey@idiap.ch>
 *
 * @brief Implements an image format reader/writer using giflib.
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
#include <gif_lib.h>
}

// (Ugly) copy of QuantizeBuffer function definition that was inlined (only) in giflib 4.2
#if defined(GIFLIB_MAJOR) && defined(GIFLIB_MINOR) && GIFLIB_MAJOR == 4 && GIFLIB_MINOR == 2

#define ABS(x)    ((x) > 0 ? (x) : (-(x)))
#define COLOR_ARRAY_SIZE 32768
#define BITS_PER_PRIM_COLOR 5
#define MAX_PRIM_COLOR 0x1f

static int SortRGBAxis;

typedef struct QuantizedColorType {
  GifByteType RGB[3];
  GifByteType NewColorIndex;
  long Count;
  struct QuantizedColorType *Pnext;
} QuantizedColorType;

typedef struct NewColorMapType {
  GifByteType RGBMin[3], RGBWidth[3];
  unsigned int NumEntries; /* # of QuantizedColorType in linked list below */
  unsigned long Count; /* Total number of pixels in all the entries */
  QuantizedColorType *QuantizedColors;
} NewColorMapType;

// Routine called by qsort to compare two entries.
static int
SortCmpRtn(const void *Entry1, const void *Entry2) 
{
  return (*((QuantizedColorType **) Entry1))->RGB[SortRGBAxis] -
    (*((QuantizedColorType **) Entry2))->RGB[SortRGBAxis];
} 

// Routine to subdivide the RGB space recursively using median cut in each
// axes alternatingly until ColorMapSize different cubes exists.
// The biggest cube in one dimension is subdivide unless it has only one entry.
// Returns GIF_ERROR if failed, otherwise GIF_OK.
static int
SubdivColorMap(NewColorMapType * NewColorSubdiv,
    unsigned int ColorMapSize,
    unsigned int *NewColorMapSize) {
  int MaxSize;
  unsigned int i, j, Index = 0, NumEntries, MinColor, MaxColor;
  long Sum, Count;
  QuantizedColorType *QuantizedColor, **SortArray;
  while (ColorMapSize > *NewColorMapSize) {
    // Find candidate for subdivision:
    MaxSize = -1;
    for (i = 0; i < *NewColorMapSize; i++) {
      for (j = 0; j < 3; j++) {
        if ((((int)NewColorSubdiv[i].RGBWidth[j]) > MaxSize) &&
            (NewColorSubdiv[i].NumEntries > 1)) {
          MaxSize = NewColorSubdiv[i].RGBWidth[j];
          Index = i;
          SortRGBAxis = j;
        }
      }
    }
    if (MaxSize == -1)
      return GIF_OK;
    // Split the entry Index into two along the axis SortRGBAxis:
    // Sort all elements in that entry along the given axis and split at
    // the median.
    SortArray = (QuantizedColorType **)malloc(
        sizeof(QuantizedColorType *) *
        NewColorSubdiv[Index].NumEntries);
    if (SortArray == NULL)
      return GIF_ERROR;
    for (j = 0, QuantizedColor = NewColorSubdiv[Index].QuantizedColors;
        j < NewColorSubdiv[Index].NumEntries && QuantizedColor != NULL;
        j++, QuantizedColor = QuantizedColor->Pnext)
      SortArray[j] = QuantizedColor;
    qsort(SortArray, NewColorSubdiv[Index].NumEntries,
        sizeof(QuantizedColorType *), SortCmpRtn);
    // Relink the sorted list into one:
    for (j = 0; j < NewColorSubdiv[Index].NumEntries - 1; j++)
      SortArray[j]->Pnext = SortArray[j + 1];
    SortArray[NewColorSubdiv[Index].NumEntries - 1]->Pnext = NULL;
    NewColorSubdiv[Index].QuantizedColors = QuantizedColor = SortArray[0];
    free((char *)SortArray);
    // Now simply add the Counts until we have half of the Count:
    Sum = NewColorSubdiv[Index].Count / 2 - QuantizedColor->Count;
    NumEntries = 1;
    Count = QuantizedColor->Count;
    while (QuantizedColor->Pnext != NULL &&
        (Sum -= QuantizedColor->Pnext->Count) >= 0 &&
        QuantizedColor->Pnext->Pnext != NULL) {
      QuantizedColor = QuantizedColor->Pnext;
      NumEntries++;
      Count += QuantizedColor->Count;
    }
    // Save the values of the last color of the first half, and first
    // of the second half so we can update the Bounding Boxes later.
    // Also as the colors are quantized and the BBoxes are full 0..255,
    // they need to be rescaled.
    MaxColor = QuantizedColor->RGB[SortRGBAxis]; // Max. of first half
    // coverity[var_deref_op]
    MinColor = QuantizedColor->Pnext->RGB[SortRGBAxis]; // of second
    MaxColor <<= (8 - BITS_PER_PRIM_COLOR);
    MinColor <<= (8 - BITS_PER_PRIM_COLOR);
    // Partition right here:
    NewColorSubdiv[*NewColorMapSize].QuantizedColors =
      QuantizedColor->Pnext;
    QuantizedColor->Pnext = NULL;
    NewColorSubdiv[*NewColorMapSize].Count = Count;
    NewColorSubdiv[Index].Count -= Count;
    NewColorSubdiv[*NewColorMapSize].NumEntries =
      NewColorSubdiv[Index].NumEntries - NumEntries;
    NewColorSubdiv[Index].NumEntries = NumEntries;
    for (j = 0; j < 3; j++) {
      NewColorSubdiv[*NewColorMapSize].RGBMin[j] =
        NewColorSubdiv[Index].RGBMin[j];
      NewColorSubdiv[*NewColorMapSize].RGBWidth[j] =
        NewColorSubdiv[Index].RGBWidth[j];
    }
    NewColorSubdiv[*NewColorMapSize].RGBWidth[SortRGBAxis] =
      NewColorSubdiv[*NewColorMapSize].RGBMin[SortRGBAxis] +
      NewColorSubdiv[*NewColorMapSize].RGBWidth[SortRGBAxis] - MinColor;
    NewColorSubdiv[*NewColorMapSize].RGBMin[SortRGBAxis] = MinColor;
    NewColorSubdiv[Index].RGBWidth[SortRGBAxis] =
      MaxColor - NewColorSubdiv[Index].RGBMin[SortRGBAxis];
    (*NewColorMapSize)++;
  }
  return GIF_OK;
}

// Quantize high resolution image into lower one. Input image consists of a
// 2D array for each of the RGB colors with size Width by Height. There is no
// Color map for the input. Output is a quantized image with 2D array of
// indexes into the output color map.
// Note input image can be 24 bits at the most (8 for red/green/blue) and
// the output has 256 colors at the most (256 entries in the color map.).
// ColorMapSize specifies size of color map up to 256 and will be updated to
// real size before returning.
// Also non of the parameter are allocated by this routine.
// This function returns GIF_OK if succesfull, GIF_ERROR otherwise.
static int 
QuantizeBuffer(unsigned int Width, unsigned int Height, int *ColorMapSize,
  GifByteType * RedInput, GifByteType * GreenInput, GifByteType * BlueInput, 
  GifByteType * OutputBuffer, GifColorType * OutputColorMap) 
{
  unsigned int Index, NumOfEntries;
  int i, j, MaxRGBError[3];
  unsigned int NewColorMapSize;
  long Red, Green, Blue;
  NewColorMapType NewColorSubdiv[256];
  QuantizedColorType *ColorArrayEntries, *QuantizedColor;
  ColorArrayEntries = (QuantizedColorType *)malloc(
      sizeof(QuantizedColorType) * COLOR_ARRAY_SIZE);
  if (ColorArrayEntries == NULL) {
    return GIF_ERROR;
  }
  for (i = 0; i < COLOR_ARRAY_SIZE; i++) {
    ColorArrayEntries[i].RGB[0] = i >> (2 * BITS_PER_PRIM_COLOR);
    ColorArrayEntries[i].RGB[1] = (i >> BITS_PER_PRIM_COLOR) &
      MAX_PRIM_COLOR;
    ColorArrayEntries[i].RGB[2] = i & MAX_PRIM_COLOR;
    ColorArrayEntries[i].Count = 0;
  }
  // Sample the colors and their distribution:
  for (i = 0; i < (int)(Width * Height); i++) {
    Index = ((RedInput[i] >> (8 - BITS_PER_PRIM_COLOR)) <<
        (2 * BITS_PER_PRIM_COLOR)) +
      ((GreenInput[i] >> (8 - BITS_PER_PRIM_COLOR)) <<
       BITS_PER_PRIM_COLOR) +
      (BlueInput[i] >> (8 - BITS_PER_PRIM_COLOR));
    ColorArrayEntries[Index].Count++;
  }
  // Put all the colors in the first entry of the color map, and call the
  // recursive subdivision process.
  for (i = 0; i < 256; i++) {
    NewColorSubdiv[i].QuantizedColors = NULL;
    NewColorSubdiv[i].Count = NewColorSubdiv[i].NumEntries = 0;
    for (j = 0; j < 3; j++) {
      NewColorSubdiv[i].RGBMin[j] = 0;
      NewColorSubdiv[i].RGBWidth[j] = 255;
    }
  }
  // Find the non empty entries in the color table and chain them:
  for (i = 0; i < COLOR_ARRAY_SIZE; i++)
    if (ColorArrayEntries[i].Count > 0)
      break;
  QuantizedColor = NewColorSubdiv[0].QuantizedColors = &ColorArrayEntries[i];
  NumOfEntries = 1;
  while (++i < COLOR_ARRAY_SIZE)
    if (ColorArrayEntries[i].Count > 0) {
      QuantizedColor->Pnext = &ColorArrayEntries[i];
      QuantizedColor = &ColorArrayEntries[i];
      NumOfEntries++;
    }
  QuantizedColor->Pnext = NULL;
  NewColorSubdiv[0].NumEntries = NumOfEntries; // Different sampled colors
  NewColorSubdiv[0].Count = ((long)Width) * Height; // Pixels
  NewColorMapSize = 1;
  if (SubdivColorMap(NewColorSubdiv, *ColorMapSize, &NewColorMapSize) !=
      GIF_OK) {
    free((char *)ColorArrayEntries);
    return GIF_ERROR;
  }
  if (NewColorMapSize < *ColorMapSize) {
    // And clear rest of color map:
    for (i = NewColorMapSize; i < *ColorMapSize; i++)
      OutputColorMap[i].Red = OutputColorMap[i].Green =
        OutputColorMap[i].Blue = 0;
  }
  // Average the colors in each entry to be the color to be used in the
  // output color map, and plug it into the output color map itself.
  for (i = 0; i < NewColorMapSize; i++) {
    if ((j = NewColorSubdiv[i].NumEntries) > 0) {
      QuantizedColor = NewColorSubdiv[i].QuantizedColors;
      Red = Green = Blue = 0;
      while (QuantizedColor) {
        QuantizedColor->NewColorIndex = i;
        Red += QuantizedColor->RGB[0];
        Green += QuantizedColor->RGB[1];
        Blue += QuantizedColor->RGB[2];
        QuantizedColor = QuantizedColor->Pnext;
      }
      OutputColorMap[i].Red = (Red << (8 - BITS_PER_PRIM_COLOR)) / j;
      OutputColorMap[i].Green = (Green << (8 - BITS_PER_PRIM_COLOR)) / j;
      OutputColorMap[i].Blue = (Blue << (8 - BITS_PER_PRIM_COLOR)) / j;
    } else
      fprintf(stderr,
          "\n: Null entry in quantized color map - that's weird.\n");
  }
  // Finally scan the input buffer again and put the mapped index in the
  // output buffer.
  MaxRGBError[0] = MaxRGBError[1] = MaxRGBError[2] = 0;
  for (i = 0; i < (int)(Width * Height); i++) {
    Index = ((RedInput[i] >> (8 - BITS_PER_PRIM_COLOR)) <<
        (2 * BITS_PER_PRIM_COLOR)) +
      ((GreenInput[i] >> (8 - BITS_PER_PRIM_COLOR)) <<
       BITS_PER_PRIM_COLOR) +
      (BlueInput[i] >> (8 - BITS_PER_PRIM_COLOR));
    Index = ColorArrayEntries[Index].NewColorIndex;
    OutputBuffer[i] = Index;
    if (MaxRGBError[0] < ABS(OutputColorMap[Index].Red - RedInput[i]))
      MaxRGBError[0] = ABS(OutputColorMap[Index].Red - RedInput[i]);
    if (MaxRGBError[1] < ABS(OutputColorMap[Index].Green - GreenInput[i]))
      MaxRGBError[1] = ABS(OutputColorMap[Index].Green - GreenInput[i]);
    if (MaxRGBError[2] < ABS(OutputColorMap[Index].Blue - BlueInput[i]))
      MaxRGBError[2] = ABS(OutputColorMap[Index].Blue - BlueInput[i]);
  }
  free((char *)ColorArrayEntries);
  *ColorMapSize = NewColorMapSize;
  return GIF_OK;
}

#undef ABS
#undef COLOR_ARRAY_SIZE
#undef BITS_PER_PRIM_COLOR
#undef MAX_PRIM_COLOR 
#endif // End of ugly QuantizeBuffer definition for giflib 4.2


static boost::shared_ptr<GifFileType> make_dfile(const char *filename)
{
#if defined(GIF_LIB_VERSION) || (GIFLIB_MAJOR < 5)
  GifFileType* fp = DGifOpenFileName(filename);
#else
  int error;
  GifFileType* fp = DGifOpenFileName(filename, &error);
#endif
  if(fp == 0) throw bob::io::FileNotReadable(filename);
  return boost::shared_ptr<GifFileType>(fp, DGifCloseFile);
}

static boost::shared_ptr<GifFileType> make_efile(const char *filename)
{
#if defined(GIF_LIB_VERSION) || (GIFLIB_MAJOR < 5)
  GifFileType* fp = EGifOpenFileName(filename, false);
#else
  int error;
  GifFileType* fp = EGifOpenFileName(filename, false, &error);
#endif
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

#if defined(GIF_LIB_VERSION) || (GIFLIB_MAJOR < 5)
  if((OutputColorMap = MakeMapObject(ColorMapSize, NULL)) == 0)
#else
  if((OutputColorMap = GifMakeMapObject(ColorMapSize, NULL)) == 0)
#endif
    throw std::runtime_error("GIF: error in GifMakeMapObject().");

#if defined(GIF_LIB_VERSION) || (GIFLIB_MAJOR < 5)
  if(QuantizeBuffer(width, height, &ColorMapSize,
       red_buffer, green_buffer, blue_buffer, output_buffer.get(), OutputColorMap->Colors) == GIF_ERROR)
#else
  if(GifQuantizeBuffer(width, height, &ColorMapSize,
       red_buffer, green_buffer, blue_buffer, output_buffer.get(), OutputColorMap->Colors) == GIF_ERROR)
#endif
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
#if defined(GIF_LIB_VERSION) || (GIFLIB_MAJOR < 5)
  FreeMapObject(OutputColorMap);
#else
  GifFreeMapObject(OutputColorMap);
#endif
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

