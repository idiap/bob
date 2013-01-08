/**
 * @file io/cxx/ImageBmpFile.cc
 * @date Wed Nov 28 15:36:00 2012 +0200
 * @author Laurent El Shafey <laurent.el-shafey@idiap.ch>
 *
 * @brief Implements an image format reader/writer for BMP files.
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
#include "bob/core/logging.h"

// The following documentation is mostly coming from wikipedia: 
// http://en.wikipedia.org/wiki/BMP_file_format

// BMP header (14 bytes)
typedef struct {
  // 1. The header field used to identify the BMP & DIB file is 0x42 0x4D in 
  //    hexadecimal, same as BM in ASCII. The following entries are possible: 
  //      BM – Windows 3.1x, 95, NT, ... etc.
  //      BA – OS/2 struct Bitmap Array
  //      CI – OS/2 struct Color Icon
  //      CP – OS/2 const Color Pointer
  //      IC – OS/2 struct Icon
  //      PT – OS/2 Pointer
  uint8_t signature[2];  
  // 2. The size of the BMP file in bytes
  uint32_t file_size;
  // 3. Reserved; actual value depends on the application that creates the image
  uint16_t reserved1;
  // 4. Reserved; actual value depends on the application that creates the image
  uint16_t reserved2;
  // 5. The offset, i.e. starting address, of the byte where the bitmap image 
  //    data (pixel array) can be found
  uint32_t offset;
} bmp_header_t;

// DIB header (bitmap information header)
// "This block of bytes tells the application detailed information about the
//  image, which will be used to display the image on the screen. The block 
//  also matches the header used internally by Windows and OS/2 and has several
//  different variants. All of them contain a dword (32 bit) field, specifying 
//  their size, so that an application can easily determine which header is 
//  used in the image."
//   - BITMAPCOREHEADER/OS21XBITMAPHEADER (12 bytes): 
//      -> OS/2 and also all Windows versions since Windows 3.0
//   - BITMAPCOREHEADER2/OS22XBITMAPHEADER (64 bytes):
//      -> OS/2
//   - BITMAPINFOHEADER (40 bytes):
//      -> all Windows versions since Windows 3.0
//   - BITMAPV2INFOHEADER (52 bytes)
//   - BITMAPV3INFOHEADER (56 bytes)
//   - BITMAPV4HEADER (108 bytes)
//      -> all Windows versions since Windows 95/NT4
//   - BITMAPV5HEADER (124 bytes)
//      -> Windows 98/2000 and newer

// We currently only support the following four DIB headers
// a/ BITMAPINFOHEADER / BITMAPV4HEADER / BITMAPV5HEADER
typedef struct {
  //// BITMAPINFOHEADER ////
  // 1. The size of this header (40 bytes)
  uint32_t header_size;
  // 2. The bitmap width in pixels (signed integer).
  int32_t width;
  // 3. The bitmap height in pixels (signed integer).
  int32_t height;
  // 4. The number of color planes being used. Must be set to 1.
  uint16_t n_planes;
  // 5. The number of bits per pixel, which is the color depth of the image.
  //    Typical values are 1, 4, 8, 16, 24 and 32.
  uint16_t depth;
  // 6. The compression method being used. See the next table for a list of 
  //    possible values.
  uint32_t compression_type;
  // 7. The image size. This is the size of the raw bitmap data (see below), 
  //    and should not be confused with the file size.
  uint32_t image_size;
  // 8. The horizontal resolution of the image. (pixel per meter, signed integer)
  int32_t hres;
  // 9. The vertical resolution of the image. (pixel per meter, signed integer)
  int32_t vres;
  // 10. The number of colors in the color palette, or 0 to default to 2^n
  uint32_t n_colors;
  // 11. The number of important colors used, or 0 when every color is 
  //     important; generally ignored.
  uint32_t n_impcolors;

  //// BITMAPV4HEADER ////
  // 12. RGBA bitmask
  uint32_t r_bitmask;
  uint32_t g_bitmask;
  uint32_t b_bitmask;
  uint32_t a_bitmask;
  // 13. Colorspace type
  uint32_t colorspace_type;
  // 14. Colorspace endpoints
  uint32_t colorspace_endpoints[9];
  // 15. Gamma for RGB channels
  uint32_t r_gamma;
  uint32_t g_gamma;
  uint32_t b_gamma;

  //// BITMAPV5HEADER ////
  // 16. Intent
  uint32_t intent;
  // 17. Profile data
  uint32_t profile_data;
  // 18. Profile size
  uint32_t profile_size;
  // 19. reserved
  uint32_t reserved;
} bmp_dib_win_header_t;

// Compression methods
typedef enum {
  BI_RGB=0, // none. Most common
  BI_RLE8, // RLE 8-bit/pixel. Can be used only with 8-bit/pixel bitmaps
  BI_RLE4, // RLE 4-bit/pixel. Can be used only with 4-bit/pixel bitmaps
  BI_BITFIELDS, //  Bit field or Huffman 1D compression for BITMAPCOREHEADER2. 
                //  Pixel format defined by bit masks or Huffman 1D compressed bitmap for BITMAPCOREHEADER2
  BI_JPEG, // JPEG or RLE-24 compression for BITMAPCOREHEADER2.
           // The bitmap contains a JPEG image or RLE-24 compressed bitmap for BITMAPCOREHEADER2
  BI_PNG, // PNG. The bitmap contains a PNG image
  BI_ALPHABITFIELDS // Bit field. This value is valid in Windows CE .NET 4.0 and later.
} bmp_compression_method;

// d/ BITMAPCOREHEADER/OS21XBITMAPHEADER
typedef struct {
  // 1. The size of this header (12 bytes)
  uint32_t header_size;
  // 2. The bitmap width in pixels.
  uint16_t width;
  // 3. The bitmap height in pixels.
  uint16_t height;
  // 4. The number of color planes being used; 1 is the only legal value.
  uint16_t n_planes;
  // 5. The number of bits per pixel, which is the color depth of the image.
  //    Typical values are 1, 4, 8 and 24.
  uint16_t depth;
} bmp_dib_os2v1_header_t;

typedef enum {
  OS2V1=0,
  WINV1=2,
  WINV4=4,
  WINV5=5
} bmp_dib_header_type;

typedef struct {
  uint32_t r_shift;
  uint32_t r_mask;
  uint32_t g_shift;
  uint32_t g_mask;
  uint32_t b_shift;
  uint32_t b_mask;
} bmp_bitmask_t;

// Container for the various DIB headers
typedef struct
{
  bmp_dib_header_type header_type;
  bool bottom_up;
  size_t height;
  size_t width;
  size_t depth;
  size_t cmap_size;
  bool has_bitmask;
  uint32_t r_bitmask;
  uint32_t g_bitmask;
  uint32_t b_bitmask;
  bmp_bitmask_t bitmask;
  union {
    bmp_dib_win_header_t win;
    bmp_dib_os2v1_header_t os2v1;
  } dib_header;
} bmp_dib_header_t;

// RGB pixel
typedef struct
{
  uint8_t r;
  uint8_t g;
  uint8_t b;
} pixel_t;


/* Commented helper functions
  Useful for debugging or to extend the support of additional bmp files in the future/

#include <bitset>

static void
bmp_print_header(bmp_header_t *hdr)
{
  std::cout << "Signature=" << hdr->signature[0] << hdr->signature[1] << std::endl;
  std::cout << "File size=" << hdr->file_size << std::endl;
  std::cout << "Reserved1=" << hdr->reserved1 << std::endl;
  std::cout << "Reserved2=" << hdr->reserved2 << std::endl;
  std::cout << "Offset=" << hdr->offset << std::endl;
}

static void
bmp_print_dib_header(bmp_dib_header_t *hdr)
{
  std::cout << "header_type=" << hdr->header_type << std::endl;
  std::cout << "bottom_up=" << hdr->bottom_up << std::endl;
  std::cout << "height=" << hdr->height << std::endl;
  std::cout << "width=" << hdr->width << std::endl;
  std::cout << "depth=" << hdr->depth << std::endl;
  std::cout << "cmap_size=" << hdr->cmap_size << std::endl;
  std::cout << "has_bitmask=" << hdr->has_bitmask << std::endl;
  std::bitset<32> r(hdr->r_bitmask);
  std::cout << " r_bitmask=" << r << std::endl;
  std::bitset<32> g(hdr->g_bitmask);
  std::cout << " g_bitmask=" << g << std::endl;
  std::bitset<32> b(hdr->b_bitmask);
  std::cout << " b_bitmask=" << b << std::endl;
  
  switch(hdr->header_type)
  {
    case WINV1:
      std::cout << "Header size=" << hdr->dib_header.win.header_size << std::endl;
      std::cout << "Width=" << hdr->dib_header.win.width << std::endl;
      std::cout << "Height=" << hdr->dib_header.win.height << std::endl;
      std::cout << "N-planes=" << hdr->dib_header.win.n_planes << std::endl;
      std::cout << "Depth=" << hdr->dib_header.win.depth << std::endl;
      std::cout << "Compresion type=" << hdr->dib_header.win.compression_type << std::endl;
      std::cout << "Image size=" << hdr->dib_header.win.image_size << std::endl;
      std::cout << "Horizontal resolution=" << hdr->dib_header.win.hres << std::endl;
      std::cout << "Vertical resolution=" << hdr->dib_header.win.vres << std::endl;
      std::cout << "N-colors=" << hdr->dib_header.win.n_colors << std::endl;
      std::cout << "N-impcolors=" << hdr->dib_header.win.n_impcolors << std::endl;
      break;
    case OS2V1:
      std::cout << "Header size=" << hdr->dib_header.os2v1.header_size << std::endl;
      std::cout << "Width=" << hdr->dib_header.os2v1.width << std::endl;
      std::cout << "Height=" << hdr->dib_header.os2v1.height << std::endl;
      std::cout << "N-planes=" << hdr->dib_header.os2v1.n_planes << std::endl;
      std::cout << "Depth=" << hdr->dib_header.os2v1.depth << std::endl;
      break;
    default:
      break;
  }
}
*/

// Read the 14 bytes header from the current FILE position
// The FILE pointer is increased of 14 bytes
static void 
bmp_read_bmp_header(FILE * const input_file, bmp_header_t *hdr)
{ 
  if(fread(&hdr->signature[0], sizeof(uint8_t), 2, input_file) != 2)
    throw std::runtime_error("bmp: error while reading bmp header (signature)");
  if(fread(&hdr->file_size, sizeof(uint32_t), 1, input_file) != 1)
    throw std::runtime_error("bmp: error while reading bmp header (file size)");
  if(fread(&hdr->reserved1, sizeof(uint16_t), 1, input_file) != 1)
    throw std::runtime_error("bmp: error while reading bmp header (reserved1)");
  if(fread(&hdr->reserved2, sizeof(uint16_t), 1, input_file) != 1)
    throw std::runtime_error("bmp: error while reading bmp header (reserved2)");
  if(fread(&hdr->offset, sizeof(uint32_t), 1, input_file) != 1)
    throw std::runtime_error("bmp: error while reading bmp header (offset)");
}

static uint32_t bmp_firstone_index(uint32_t v)
{
  if(v==0)
    throw std::runtime_error("bmp: bmp_firstone_index (used by the bitmask parser) can not deal with 0 values.");
  uint32_t pos = 0;
  while((v % 2) == 0)
  {
    v >>= 1;
    ++pos;
  }
  return pos;
}

static uint32_t bmp_lastone_index(uint32_t v)
{
  if(v==0)
    throw std::runtime_error("bmp: bmp_lastone_index (used by the bitmask parser) can not deal with 0 values.");
  uint32_t pos = 0;
  while(v != 1)
  {
    v >>= 1;
    ++pos;
  }
  return pos;
}

// Update the bitmask structure by parsing the bitfields
static void bmp_update_bitmask_structure(uint32_t r, uint32_t g, uint32_t b, bmp_bitmask_t *bm)
{
  // Shift
  bm->r_shift = bmp_firstone_index(r);
  bm->g_shift = bmp_firstone_index(g);
  bm->b_shift = bmp_firstone_index(b);
  
  // Mask
  bm->r_mask =  (1 << (bmp_lastone_index(r) - bm->r_shift + 1)) -1;
  bm->g_mask =  (1 << (bmp_lastone_index(g) - bm->g_shift + 1)) -1;
  bm->b_mask =  (1 << (bmp_lastone_index(b) - bm->b_shift + 1)) -1;
}


// Read the and parse the windows DIB header bitmasks from the current FILE position
static void bmp_read_bitmask_win_dib_header(FILE * const input_file, bmp_dib_header_t *dib_hdr)
{
  dib_hdr->has_bitmask = true;
  if(fread(&dib_hdr->r_bitmask, sizeof(uint32_t), 1, input_file) != 1)
    throw std::runtime_error("bmp: error while reading bmp DIB header (Red bitmask)");
  if(fread(&dib_hdr->g_bitmask, sizeof(uint32_t), 1, input_file) != 1)
    throw std::runtime_error("bmp: error while reading bmp DIB header (Green bitmask)");
  if(fread(&dib_hdr->b_bitmask, sizeof(uint32_t), 1, input_file) != 1)
    throw std::runtime_error("bmp: error while reading bmp DIB header (Blue bitmask)");
  bmp_update_bitmask_structure(dib_hdr->r_bitmask, dib_hdr->g_bitmask, dib_hdr->b_bitmask, &dib_hdr->bitmask);
}

// Read the Winv1 DIB header from the current FILE position
// The FILE pointer is increased according to the size of the DIB header 
//  (if DIB type is supported)
static void 
bmp_read_winv1_dib_header(FILE * const input_file, bmp_dib_header_t *dib_hdr, const bool winv1=true)
{
  if(fread(&dib_hdr->dib_header.win.width, sizeof(int32_t), 1, input_file) != 1)
    throw std::runtime_error("bmp: error while reading bmp DIB header (width)");
  if(fread(&dib_hdr->dib_header.win.height, sizeof(int32_t), 1, input_file) != 1)
    throw std::runtime_error("bmp: error while reading bmp DIB header (height)");
  if(fread(&dib_hdr->dib_header.win.n_planes, sizeof(uint16_t), 1, input_file) != 1)
    throw std::runtime_error("bmp: error while reading bmp DIB header (number of planes)");
  if(fread(&dib_hdr->dib_header.win.depth, sizeof(uint16_t), 1, input_file) != 1)
    throw std::runtime_error("bmp: error while reading bmp DIB header (depth)");
  if(fread(&dib_hdr->dib_header.win.compression_type, sizeof(uint32_t), 1, input_file) != 1)
    throw std::runtime_error("bmp: error while reading bmp DIB header (compression type)");
  if(dib_hdr->dib_header.win.compression_type != BI_RGB && 
     dib_hdr->dib_header.win.compression_type != BI_BITFIELDS)
    throw std::runtime_error("bmp: unsupported compression type in header");
  if(fread(&dib_hdr->dib_header.win.image_size, sizeof(uint32_t), 1, input_file) != 1)
    throw std::runtime_error("bmp: error while reading bmp DIB header (image size)");
  if(fread(&dib_hdr->dib_header.win.hres, sizeof(int32_t), 1, input_file) != 1)
    throw std::runtime_error("bmp: error while reading bmp DIB header (horizontal resolution)");
  if(fread(&dib_hdr->dib_header.win.vres, sizeof(int32_t), 1, input_file) != 1)
    throw std::runtime_error("bmp: error while reading bmp DIB header (vertical resolution)");
  if(fread(&dib_hdr->dib_header.win.n_colors, sizeof(uint32_t), 1, input_file) != 1)
    throw std::runtime_error("bmp: error while reading bmp DIB header (number of colors)");
  if(fread(&dib_hdr->dib_header.win.n_impcolors, sizeof(uint32_t), 1, input_file) != 1)
    throw std::runtime_error("bmp: error while reading bmp DIB header (number of important colors)");

  // Update "standard" DIB attributes 
  dib_hdr->bottom_up = (dib_hdr->dib_header.win.height > 0);
  dib_hdr->height = (dib_hdr->dib_header.win.height > 0 ? dib_hdr->dib_header.win.height : -dib_hdr->dib_header.win.height);
  dib_hdr->width = (dib_hdr->dib_header.win.width > 0 ? dib_hdr->dib_header.win.width : -dib_hdr->dib_header.win.width);
  dib_hdr->depth = dib_hdr->dib_header.win.depth;

  // Update color map size attribute
  if(dib_hdr->depth <= 8) 
  {
    uint16_t n_colors = dib_hdr->dib_header.win.n_colors;
    if(n_colors != 0) {
      if(n_colors > (1 << dib_hdr->depth))
        throw std::runtime_error("bmp: error while reading bmp DIB header (Colormap).");
      else
        dib_hdr->cmap_size = n_colors;
    } 
    else
      dib_hdr->cmap_size = (1 << dib_hdr->depth);
  } 
  else if (dib_hdr->depth == 24 || dib_hdr->depth == 16 || dib_hdr->depth == 32)
    dib_hdr->cmap_size = 0;
  else
    throw std::runtime_error("bmp: error while reading bmp DIB header (Colormap: Unrecognized bits per pixel in Windows BMP file header).");

  // If BIT_FIELD COMPRESSION_TYPE is set, we need to read the bitmasks
  if(winv1 && dib_hdr->dib_header.win.compression_type == BI_BITFIELDS)
    bmp_read_bitmask_win_dib_header(input_file, dib_hdr);
  else
    dib_hdr->has_bitmask = false; 
}

// Read the Winv4 DIB header part from the current FILE position
// The FILE pointer is increased according to the size of the DIB header 
//  (if DIB type is supported)
static void 
bmp_read_winv4_dib_header(FILE * const input_file, bmp_dib_header_t *dib_hdr)
{
  // 1. RGBA bitmask
  bmp_read_bitmask_win_dib_header(input_file, dib_hdr);
  dib_hdr->dib_header.win.r_bitmask = dib_hdr->r_bitmask;
  dib_hdr->dib_header.win.g_bitmask = dib_hdr->g_bitmask;
  dib_hdr->dib_header.win.b_bitmask = dib_hdr->b_bitmask;
  if(fread(&dib_hdr->dib_header.win.a_bitmask, sizeof(uint32_t), 1, input_file) != 1)
    throw std::runtime_error("bmp: error while reading bmp DIB header (Alpha bitmask)");
  // 2. Colorspace type
  if(fread(&dib_hdr->dib_header.win.colorspace_type, sizeof(uint32_t), 1, input_file) != 1)
    throw std::runtime_error("bmp: error while reading bmp DIB header (Colorspace type)");
  // 3. Colorspace endpoints
  if(fread(&dib_hdr->dib_header.win.colorspace_endpoints, sizeof(uint32_t), 9, input_file) != 9)
    throw std::runtime_error("bmp: error while reading bmp DIB header (Colorspace endpoints)");
  // 4. Gamma RGB channels
  if(fread(&dib_hdr->dib_header.win.r_gamma, sizeof(uint32_t), 1, input_file) != 1)
    throw std::runtime_error("bmp: error while reading bmp DIB header (Gamma red channel)");
  if(fread(&dib_hdr->dib_header.win.g_gamma, sizeof(uint32_t), 1, input_file) != 1)
    throw std::runtime_error("bmp: error while reading bmp DIB header (Gamma green channel)");
  if(fread(&dib_hdr->dib_header.win.b_gamma, sizeof(uint32_t), 1, input_file) != 1)
    throw std::runtime_error("bmp: error while reading bmp DIB header (Gamma blue channel)");
}

// Read the Winv5 DIB header part from the current FILE position
// The FILE pointer is increased according to the size of the DIB header 
//  (if DIB type is supported)
static void 
bmp_read_winv5_dib_header(FILE * const input_file, bmp_dib_header_t *dib_hdr)
{
  // 1. Intent
  if(fread(&dib_hdr->dib_header.win.intent, sizeof(uint32_t), 1, input_file) != 1)
    throw std::runtime_error("bmp: error while reading bmp DIB header (Intent)");
  // 2. Profile data
  if(fread(&dib_hdr->dib_header.win.profile_data, sizeof(uint32_t), 1, input_file) != 1)
    throw std::runtime_error("bmp: error while reading bmp DIB header (Profile data)");
  // 3. Profile size
  if(fread(&dib_hdr->dib_header.win.profile_size, sizeof(uint32_t), 1, input_file) != 1)
    throw std::runtime_error("bmp: error while reading bmp DIB header (Profile size)");
  // 4. Reserved
  if(fread(&dib_hdr->dib_header.win.reserved, sizeof(uint32_t), 1, input_file) != 1)
    throw std::runtime_error("bmp: error while reading bmp DIB header (Reserved)");
}

// Read the OS2v1 DIB header from the current FILE position
// The FILE pointer is increased according to the size of the DIB header 
//  (if DIB type is supported)
static void 
bmp_read_os2v1_dib_header(FILE * const input_file, bmp_dib_header_t *dib_hdr)
{
  // Read the OS2v1 DIB header
  if(fread(&dib_hdr->dib_header.os2v1.width, sizeof(uint16_t), 1, input_file) != 1)
    throw std::runtime_error("bmp: error while reading bmp DIB header (width)");
  if(fread(&dib_hdr->dib_header.os2v1.height, sizeof(uint16_t), 1, input_file) != 1)
    throw std::runtime_error("bmp: error while reading bmp DIB header (height)");
  if(fread(&dib_hdr->dib_header.os2v1.n_planes, sizeof(uint16_t), 1, input_file) != 1)
    throw std::runtime_error("bmp: error while reading bmp DIB header (number of planes)");
  if(fread(&dib_hdr->dib_header.os2v1.depth, sizeof(uint16_t), 1, input_file) != 1)
    throw std::runtime_error("bmp: error while reading bmp DIB header (depth)");

  // Update "standard" DIB attributes 
  dib_hdr->bottom_up = true;
  dib_hdr->height = dib_hdr->dib_header.os2v1.height;
  dib_hdr->width = dib_hdr->dib_header.os2v1.width;
  dib_hdr->depth = dib_hdr->dib_header.os2v1.depth;

  // Update color map size attribute
  if(dib_hdr->depth <= 8)
    dib_hdr->cmap_size = (1 << dib_hdr->depth);
  else if(dib_hdr->depth == 24)
    dib_hdr->cmap_size = 0;
  else
    throw std::runtime_error("bmp: error while reading bmp DIB header (Colormap: Unrecognized bits per pixel in OS2 BMP file header).");
}

// Read the DIB header from the current FILE position
// The FILE pointer is increased according to the size of the DIB header 
//  (if DIB type is supported)
static void 
bmp_read_dib_header(FILE * const input_file, bmp_dib_header_t *dib_hdr)
{
  uint32_t dib_hdr_size;
  if(fread(&dib_hdr_size, sizeof(uint32_t), 1, input_file) != 1)
    throw std::runtime_error("bmp: error while reading DIB bmp header (header size)");

  // Set the DIB header type according to the read value
  switch(dib_hdr_size)
  {
    case 12: // OS2 V1
      dib_hdr->dib_header.os2v1.header_size = dib_hdr_size;
      dib_hdr->header_type = OS2V1;
      break;
    case 40: // Windows V1
      dib_hdr->dib_header.win.header_size = dib_hdr_size;
      dib_hdr->header_type = WINV1;
      break;
    case 108: // Windows V4
      dib_hdr->dib_header.win.header_size = dib_hdr_size;
      dib_hdr->header_type = WINV4;
      break;
    case 124: // Windows V5
      dib_hdr->dib_header.win.header_size = dib_hdr_size;
      dib_hdr->header_type = WINV5;
      break;
    default:
      throw std::runtime_error("bmp: Unsupported bmp file (DIB header type unsupported).");
  }

  // Read the remaining of the DIB header
  switch(dib_hdr->header_type)
  {
    case WINV1:
      // Read the windows WINV1 DIB header
      bmp_read_winv1_dib_header(input_file, dib_hdr);
      break;
    
    case WINV4:
      // Read the windows WINV1 DIB header part
      bmp_read_winv1_dib_header(input_file, dib_hdr, false);
      // Read the windows WINV4 DIB header part
      bmp_read_winv4_dib_header(input_file, dib_hdr);
      break;

    case WINV5:
      // Read the windows WINV1 DIB header part
      bmp_read_winv1_dib_header(input_file, dib_hdr, false);
      // Read the windows WINV4 DIB header part
      bmp_read_winv4_dib_header(input_file, dib_hdr);
      // Read the windows WINV5 DIB header part
      bmp_read_winv5_dib_header(input_file, dib_hdr);
      break;

    case OS2V1:
      // Read the OS2v1 DIB header
      bmp_read_os2v1_dib_header(input_file, dib_hdr);
      break;

    default:
      break;
  }
}

// Read the colormap
static void 
bmp_read_colormap(FILE * const input_file, pixel_t *color_map, size_t cmap_size, bmp_dib_header_type hdr_type)
{
  for(size_t i=0; i<cmap_size; ++i) 
  {
    /* From Netpbm: There is a document that says the bytes are ordered R,G,B,Z,
       but in practice it appears to be the following instead:
     */
    uint8_t r, g, b;

    size_t s_el = sizeof(uint8_t);
    if(fread(&b, s_el, 1, input_file) != 1)
      throw std::runtime_error("bmp: error while reading color map");
    if(fread(&g, s_el, 1, input_file) != 1)
      throw std::runtime_error("bmp: error while reading color map");
    if(fread(&r, s_el, 1, input_file) != 1)
      throw std::runtime_error("bmp: error while reading color map");

    color_map[i].r = r;
    color_map[i].g = g;
    color_map[i].b = b;

    if(hdr_type == WINV1)
    {
      if(fread(&r, s_el, 1, input_file) != 1)
        throw std::runtime_error("bmp: error while reading color map");
    }
  }    
}

// Allocate buffer for raster
static size_t bmp_get_nbytes_per_row(const bmp_dib_header_t *dib_hdr)
{
  return (dib_hdr->width * dib_hdr->depth + 31) / 32 * 4;
}

// Read the raster data
static void 
bmp_read_raster(FILE * const input_file, const bmp_dib_header_t *dib_hdr, size_t n_bytes_per_row, uint8_t *data)
{
  if(dib_hdr->bottom_up)
    for(size_t i=0; i<dib_hdr->height; ++i) 
    {
      if(fread(&data[(dib_hdr->height-1-i)*n_bytes_per_row], 1, n_bytes_per_row, input_file) != n_bytes_per_row)
        throw std::runtime_error("bmp: error while reading raster data");
    }
  else
    for(size_t i=0; i<dib_hdr->height; ++i) 
    {
      if(fread(&data[i*n_bytes_per_row], 1, n_bytes_per_row, input_file) != n_bytes_per_row)
        throw std::runtime_error("bmp: error while reading raster data");
    }
}

static boost::shared_ptr<std::FILE> make_cfile(const char *filename, const char *flags)
{
  std::FILE* fp = std::fopen(filename, flags);
  if(fp == 0) throw bob::io::FileNotReadable(filename);
  return boost::shared_ptr<std::FILE>(fp, std::fclose);
}

/**
 * LOADING
 */
static void im_peek(const std::string& path, bob::core::array::typeinfo& info) {
  // 1. BMP structures
  bmp_header_t bmp_hdr;
  bmp_dib_header_t bmp_dib_hdr;
  
  // 2. BMP file opening
  boost::shared_ptr<std::FILE> in_file = make_cfile(path.c_str(), "rb");

  // 3. Read headers
  bmp_read_bmp_header(in_file.get(), &bmp_hdr);
  bmp_read_dib_header(in_file.get(), &bmp_dib_hdr);

  // 4. Read color map
  boost::shared_array<pixel_t> cmap(new pixel_t[bmp_dib_hdr.cmap_size]);
  bmp_read_colormap(in_file.get(), cmap.get(), bmp_dib_hdr.cmap_size, bmp_dib_hdr.header_type);

  if(ftell(in_file.get()) != (long)bmp_hdr.offset)
    throw std::runtime_error("bmp: error while parsing bmp header (current file position does not match the offset value indicating where the data is stored)");

  // 5.  Set depth and number of dimensions
  info.dtype = bob::core::array::t_uint8;
  info.nd = 3;
  info.shape[0] = 3;
  info.shape[1] = bmp_dib_hdr.height;
  info.shape[2] = bmp_dib_hdr.width;
  info.update_strides();
}

static void im_load(const std::string& filename, bob::core::array::interface& b) {
  // 1. BMP structures
  bmp_header_t bmp_hdr;
  bmp_dib_header_t bmp_dib_hdr;
  
  // 2. BMP file opening
  boost::shared_ptr<std::FILE> in_file = make_cfile(filename.c_str(), "rb");

  // 3. Read headers
  bmp_read_bmp_header(in_file.get(), &bmp_hdr);
  bmp_read_dib_header(in_file.get(), &bmp_dib_hdr);

  // 4. Read color map
  boost::shared_array<pixel_t> cmap(new pixel_t[bmp_dib_hdr.cmap_size]);
  bmp_read_colormap(in_file.get(), cmap.get(), bmp_dib_hdr.cmap_size, bmp_dib_hdr.header_type);

  // 5. Read data
  size_t n_bytes_per_row = bmp_get_nbytes_per_row( &bmp_dib_hdr);
  boost::shared_array<uint8_t> rasterdata(new uint8_t[n_bytes_per_row*bmp_dib_hdr.height]);
  bmp_read_raster(in_file.get(), &bmp_dib_hdr, n_bytes_per_row, rasterdata.get());

  // 6. Convert data using the color map and put it in the RGB buffer
  const bob::core::array::typeinfo& info = b.type(); 
  long unsigned int frame_size = info.shape[1] * info.shape[2]; 
  uint8_t *element_r = static_cast<uint8_t*>(b.ptr());
  uint8_t *element_g = element_r+frame_size;
  uint8_t *element_b = element_g+frame_size;

  if(bmp_dib_hdr.depth == 24)
  {
    if(bmp_dib_hdr.has_bitmask)
    {
      for(size_t i=0; i<bmp_dib_hdr.height; ++i)
      {
        for(size_t j=0; j<bmp_dib_hdr.width; ++j)
        {
          uint32_t v = rasterdata[i*n_bytes_per_row+j*3+2] << 16 | rasterdata[i*n_bytes_per_row+j*3+1] << 8 | rasterdata[i*n_bytes_per_row+j*3];
          *element_b++ = ((v >> bmp_dib_hdr.bitmask.b_shift) & bmp_dib_hdr.bitmask.b_mask) * 255 / bmp_dib_hdr.bitmask.b_mask;
          *element_g++ = ((v >> bmp_dib_hdr.bitmask.g_shift) & bmp_dib_hdr.bitmask.g_mask) * 255 / bmp_dib_hdr.bitmask.g_mask;
          *element_r++ = ((v >> bmp_dib_hdr.bitmask.r_shift) & bmp_dib_hdr.bitmask.r_mask) * 255 / bmp_dib_hdr.bitmask.r_mask;
        }
      }
    }
    else
    {
      for(size_t i=0; i<bmp_dib_hdr.height; ++i)
      {
        for(size_t j=0; j<bmp_dib_hdr.width; ++j)
        {
          *element_b++ = rasterdata[i*n_bytes_per_row+j*3];
          *element_g++ = rasterdata[i*n_bytes_per_row+j*3+1];
          *element_r++ = rasterdata[i*n_bytes_per_row+j*3+2];
        }
      }
    }
  }
  else if(bmp_dib_hdr.depth == 16)
  {
    if(bmp_dib_hdr.has_bitmask)
    {
      for(size_t i=0; i<bmp_dib_hdr.height; ++i)
      {
        for(size_t j=0; j<bmp_dib_hdr.width; ++j)
        {
          uint16_t v =  rasterdata[i*n_bytes_per_row+j*2+1] << 8 | rasterdata[i*n_bytes_per_row+j*2];
          *element_b++ = ((v >> bmp_dib_hdr.bitmask.b_shift) & bmp_dib_hdr.bitmask.b_mask) * 255 / bmp_dib_hdr.bitmask.b_mask;
          *element_g++ = ((v >> bmp_dib_hdr.bitmask.g_shift) & bmp_dib_hdr.bitmask.g_mask) * 255 / bmp_dib_hdr.bitmask.g_mask;
          *element_r++ = ((v >> bmp_dib_hdr.bitmask.r_shift) & bmp_dib_hdr.bitmask.r_mask) * 255 / bmp_dib_hdr.bitmask.r_mask;
        }
      }
    }
    else
    {
      // Assumes 555 16 bits image by default
      for(size_t i=0; i<bmp_dib_hdr.height; ++i)
      {
        for(size_t j=0; j<bmp_dib_hdr.width; ++j)
        {
          uint16_t v =  rasterdata[i*n_bytes_per_row+j*2+1] << 8 | rasterdata[i*n_bytes_per_row+j*2];
          *element_b++ = ((v >> 0) & 0x1F) * 255 / 0x1F;
          *element_g++ = ((v >> 5) & 0x1F) * 255 / 0x1F;
          *element_r++ = ((v >> 10) & 0x1F) * 255 / 0x1F;
        }
      }
    }
  }
  else if(bmp_dib_hdr.depth == 32)
  {
    if(bmp_dib_hdr.has_bitmask)
    {
      for(size_t i=0; i<bmp_dib_hdr.height; ++i)
      {
        for(size_t j=0; j<bmp_dib_hdr.width; ++j)
        {
          uint32_t v = rasterdata[i*n_bytes_per_row+j*4+2] << 16 | rasterdata[i*n_bytes_per_row+j*4+1] << 8 | rasterdata[i*n_bytes_per_row+j*4];
          *element_b++ = ((v >> bmp_dib_hdr.bitmask.b_shift) & bmp_dib_hdr.bitmask.b_mask) * 255 / bmp_dib_hdr.bitmask.b_mask;
          *element_g++ = ((v >> bmp_dib_hdr.bitmask.g_shift) & bmp_dib_hdr.bitmask.g_mask) * 255 / bmp_dib_hdr.bitmask.g_mask;
          *element_r++ = ((v >> bmp_dib_hdr.bitmask.r_shift) & bmp_dib_hdr.bitmask.r_mask) * 255 / bmp_dib_hdr.bitmask.r_mask;
        }
      }
    }
    else
    {
      for(size_t i=0; i<bmp_dib_hdr.height; ++i)
      {
        for(size_t j=0; j<bmp_dib_hdr.width; ++j)
        {
          *element_b++ = rasterdata[i*n_bytes_per_row+j*4];
          *element_g++ = rasterdata[i*n_bytes_per_row+j*4+1];
          *element_r++ = rasterdata[i*n_bytes_per_row+j*4+2];
        }
      }
    }
  }
  else if(bmp_dib_hdr.depth == 8) 
  {
    if(bmp_dib_hdr.has_bitmask)
      throw std::runtime_error("bmp: usage of bitfields is currently restricted to 16bits depth images."); 
    pixel_t v;
    for(size_t i=0; i<bmp_dib_hdr.height; ++i)
    {
      for(size_t j=0; j<bmp_dib_hdr.width; ++j)
      {
        v = cmap[rasterdata[i*n_bytes_per_row+j]];
        *element_b++ = v.b;
        *element_g++ = v.g;
        *element_r++ = v.r;
      }
    }
  } 
  else if(bmp_dib_hdr.depth < 8) 
  {
    if(bmp_dib_hdr.has_bitmask)
      throw std::runtime_error("bmp: usage of bitfields is currently restricted to 16bits depth images."); 
    // It's a bit field color index
    const uint8_t mask = (1 << bmp_dib_hdr.depth) - 1;
    pixel_t v;
    for(size_t i=0; i<bmp_dib_hdr.height; ++i)
    {
      for(size_t j=0; j<bmp_dib_hdr.width; ++j)
      {
        const unsigned int cursor = (j*bmp_dib_hdr.depth)/8;
        const unsigned int shift = 8 - ((j*bmp_dib_hdr.depth) % 8) - bmp_dib_hdr.depth;
        const unsigned int index = (rasterdata[i*n_bytes_per_row+cursor] & (mask << shift)) >> shift;
        v = cmap[index];
        *element_b++ = v.b;
        *element_g++ = v.g;
        *element_r++ = v.r;
      }
    }
  }
}

/**
 * SAVING
 */
static void bmp_write_header(FILE * out_file, size_t file_size, size_t offset)
{
  // Signature
  static uint8_t signature[] = {'B', 'M'};
  if(fwrite(signature, sizeof(uint8_t), 2, out_file) != 2)
    throw std::runtime_error("bmp: error while writing bmp header (signature)");
  uint32_t v32;
  // File size
  v32 = file_size;
  if(fwrite(&v32, sizeof(uint32_t), 1, out_file) != 1)
    throw std::runtime_error("bmp: error while writing bmp header (file size)");
  uint16_t v16;
  // Reserved 1 and 2
  v16 = 0;
  if(fwrite(&v16, sizeof(uint16_t), 1, out_file) != 1)
    throw std::runtime_error("bmp: error while writing bmp header (reserved1)");
  if(fwrite(&v16, sizeof(uint16_t), 1, out_file) != 1)
    throw std::runtime_error("bmp: error while writing bmp header (reserved2)");
  // Offset
  v32 = offset;
  if(fwrite(&v32, sizeof(uint32_t), 1, out_file) != 1)
    throw std::runtime_error("bmp: error while writing bmp header (offset)");
}

static void bmp_write_dib_header(FILE * out_file, size_t height, size_t width)
{
  uint32_t v32;
  // 1. DIB Header size
  v32 = 40;
  if(fwrite(&v32, sizeof(uint32_t), 1, out_file) != 1)
    throw std::runtime_error("bmp: error while writing bmp DIB header (header size)");
  int32_t s32;
  // 2. The bitmap width in pixels (signed integer).
  s32 = width;
  if(fwrite(&s32, sizeof(int32_t), 1, out_file) != 1)
    throw std::runtime_error("bmp: error while writing bmp DIB header (width)");
  // 3. The bitmap height in pixels (signed integer).
  s32 = height;
  if(fwrite(&s32, sizeof(int32_t), 1, out_file) != 1)
    throw std::runtime_error("bmp: error while writing bmp DIB header (height)");
  uint16_t v16;
  // 4. The number of color planes being used. Must be set to 1.
  v16 = 1;
  if(fwrite(&v16, sizeof(uint16_t), 1, out_file) != 1)
    throw std::runtime_error("bmp: error while writing bmp DIB header (number of planes)");
  // 5. The number of bits per pixel, which is the color depth of the image.
  //    Typical values are 1, 4, 8, 16, 24 and 32.
  v16 = 24;
  if(fwrite(&v16, sizeof(uint16_t), 1, out_file) != 1)
    throw std::runtime_error("bmp: error while writing bmp DIB header (number of bits per pixel)");
  // 6. The compression method being used. See the next table for a list of 
  //    possible values.
  v32 = 0; // No compression
  if(fwrite(&v32, sizeof(uint32_t), 1, out_file) != 1)
    throw std::runtime_error("bmp: error while writing bmp DIB header (compression type)");
  // 7. The image size. This is the size of the raw bitmap data (see below), 
  //    and should not be confused with the file size.
  v32 = height * width;
  if(fwrite(&v32, sizeof(uint32_t), 1, out_file) != 1)
    throw std::runtime_error("bmp: error while writing bmp DIB header (image size)");
  // 8. The horizontal resolution of the image. (pixel per meter, signed integer)
  s32 = 3780;
  if(fwrite(&s32, sizeof(int32_t), 1, out_file) != 1)
    throw std::runtime_error("bmp: error while writing bmp DIB header (horizontal resolution)");
  // 9. The vertical resolution of the image. (pixel per meter, signed integer)
  if(fwrite(&s32, sizeof(int32_t), 1, out_file) != 1)
    throw std::runtime_error("bmp: error while writing bmp DIB header (vertical resolution)");
  // 10. The number of colors in the color palette, or 0 to default to 2^n
  v32 = 0;
  if(fwrite(&v32, sizeof(uint32_t), 1, out_file) != 1)
    throw std::runtime_error("bmp: error while writing bmp DIB header (number of colors)");
  // 11. The number of important colors used, or 0 when every color is 
  //     important; generally ignored.
  if(fwrite(&v32, sizeof(uint32_t), 1, out_file) != 1)
    throw std::runtime_error("bmp: error while writing bmp DIB header (number of important colors)");
}

// Save images in Windows V1 format with a 24 bits depth (without color map)
static void im_save_color(const bob::core::array::interface& b, FILE * out_file) 
{
  const bob::core::array::typeinfo& info = b.type();

  size_t height = info.shape[1];
  size_t width = info.shape[2];
  size_t frame_size = height * width;
  // The number of bytes per row in a bitmap file should be aligned to 4 bytes
  size_t bytes_per_row = 3 * width;
  size_t offset_per_row = (bytes_per_row % 4 ? 4 - (bytes_per_row % 4) : 0);
  bytes_per_row += offset_per_row;
  size_t image_size = height * bytes_per_row; // size without header

  const uint8_t *element_r = static_cast<const uint8_t*>(b.ptr());
  const uint8_t *element_g = element_r + frame_size;
  const uint8_t *element_b = element_g + frame_size;

  // Write headers
  size_t file_size = image_size + 54;
  bmp_write_header(out_file, file_size, 54);
  bmp_write_dib_header(out_file, height, width);

  // Write data
  uint8_t zero = 0;
  for(size_t i=0; i<height; ++i)
  {
    for(size_t j=0; j<width; ++j)
    {
      if(fwrite(&element_b[(height-1-i)*width+j], sizeof(uint8_t), 1, out_file) != 1)
        throw std::runtime_error("bmp: error while writing bmp raster data");
      if(fwrite(&element_g[(height-1-i)*width+j], sizeof(uint8_t), 1, out_file) != 1)
        throw std::runtime_error("bmp: error while writing bmp raster data");
      if(fwrite(&element_r[(height-1-i)*width+j], sizeof(uint8_t), 1, out_file) != 1)
        throw std::runtime_error("bmp: error while writing bmp raster data");
    }
    for(size_t j=0; j<offset_per_row; ++j)
      if(fwrite(&zero, sizeof(uint8_t), 1, out_file) != 1)
        throw std::runtime_error("bmp: error while writing bmp raster data");
      
  }
}

static void im_save(const std::string& filename, const bob::core::array::interface& array) {
  const bob::core::array::typeinfo& info = array.type();

  // 1. BMP file opening
  boost::shared_ptr<std::FILE> out_file = make_cfile(filename.c_str(), "wb");

  // 2. Write image
  if(info.dtype == bob::core::array::t_uint8) {
    if(info.nd == 3) {
      if(info.shape[0] != 3) throw std::runtime_error("color image does not have 3 planes on 1st. dimension");
      im_save_color(array, out_file.get());
    }
    else throw bob::io::ImageUnsupportedDimension(info.nd); 
  }
  else throw bob::io::ImageUnsupportedType(info.dtype);
}

class ImageBmpFile: public bob::io::File {

  public: //api

    ImageBmpFile(const std::string& path, char mode):
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

    virtual ~ImageBmpFile() { }

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

std::string ImageBmpFile::s_codecname = "bob.image_bmp";

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
  return boost::make_shared<ImageBmpFile>(path, mode);
}

/**
 * Takes care of codec registration per se.
 */
static bool register_codec() {
  boost::shared_ptr<bob::io::CodecRegistry> instance =
    bob::io::CodecRegistry::instance();

  instance->registerExtension(".bmp", "BMP (bob codec!)", &make_file);

  return true;
}

static bool codec_registered = register_codec();

