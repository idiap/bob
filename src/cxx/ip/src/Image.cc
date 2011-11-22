/**
 * @file cxx/ip/src/Image.cc
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
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
#include "ip/Image.h"
#include "ip/OldColor.h"
#include "ip/vision.h"

namespace Torch {

  Image::Image(int width, int height, int n_planes)
    :	ShortTensor(height, width, n_planes),
    m_setPixelCallback(setPixel1DChar)
  {
    this->resize(width, height, n_planes);
  }

  Image::~Image()
  {
  }

  // Resize the image (to new dimensions, no of planes and storage type)
  bool Image::resize(int width, int height, int n_planes)
  {
    // Check parameters
    if (	width < 1 || height < 1 || (n_planes != 1 && n_planes != 3)) {
      // Suport only images with one (gray) or 3 (RGB like) components
      Torch::message("Torch::Image::resize - invalid parameters!\n");
      return false;
    }

    // Resize only if needed
    if (getWidth() != width || getHeight() != height) {
      ShortTensor::resize(height, width, n_planes);
      ShortTensor::fill(0);
    }

    // Sets the callback to change the pixels - AA: please note this has to be
    // done irrespective of the image changing size. The only change that needs
    // to be considered is if the number of planes changes. 
    if (n_planes == 1) m_setPixelCallback = setPixel1DChar;
    else m_setPixelCallback = setPixel3DChar;

    // OK
    return true;
  }

  // Copy from some 3D tensor (should have the same dimension)
  bool Image::copyFrom(const Tensor& data)
  {
    // Check parameters
    if (data.nDimension() != nDimension()) {
      Torch::message("Torch::Image::copyFrom - invalid parameters!\n");
      return false;
    }
    for (int i = 0; i < data.nDimension(); i ++)
      if (data.size(i) != size(i)) {
        Torch::message("Torch::Image::copyFrom - invalid parameters!\n");
        return false;
      }

    // OK, copy it
    copy(&data);
    return true;
  }

  // Copy from another image (should have the same dimension)
  bool Image::copyFrom(const Image& image)
  {
    // Check parameters
    if ( getWidth() != image.getWidth() ||
        getHeight() != image.getHeight()) {
      Torch::message("Torch::Image::copyFrom - invalid image!\n");
      return false;
    }

    // Copy the image
    const int w = getWidth();
    const int h = getHeight();
    if (getNPlanes() == image.getNPlanes()) {
      // The same number of planes!
      ShortTensor::copy(&image);
    }
    else if (getNPlanes() == 1) {
      // RGB to gray
      for (int y = 0; y < h; y ++)
        for (int x = 0; x < w; x ++) {
          (*this)(y, x, 0) = rgb_to_gray(image(y, x, 0), 
              image(y, x, 1), image(y, x, 2));
        }
    }
    else if (getNPlanes() == 3) { // gray to RGB
      for (int y = 0; y < h; y ++)
        for (int x = 0; x < w; x ++) {
          const short gray = image(y, x, 0);
          (*this)(y, x, 0) = gray;
          (*this)(y, x, 1) = gray;
          (*this)(y, x, 2) = gray;
        }
    }

    return true;
  }

  // Fills an image object from a pixmap
  void Image::fillImage(const unsigned char* pixmap, int n_planes_pixmap, Image& image)
  {
    const int width = image.getWidth();
    const int height = image.getHeight();

    // Grayscale image
    if (image.getNPlanes() == 1) {
      // Grayscale pixmap
      if (n_planes_pixmap == 1) {
        for (int j = 0; j < height; j ++)
          for (int i = 0; i < width; i ++) {
            image(j, i, 0) = (short)*(pixmap ++);
          }
      }

      // RGB pixmap
      else {
        for (int j = 0; j < height; j ++)
          for (int i = 0; i < width; i ++) {
            const unsigned char r = *(pixmap ++);
            const unsigned char g = *(pixmap ++);
            const unsigned char b = *(pixmap ++);
            image(j, i, 0) = (short)rgb_to_gray(r, g, b);
          }
      }
    }

    // RGB image
    else if (image.getNPlanes() == 3) { // Grayscale pixmap
      if (n_planes_pixmap == 1) {
        for (int j = 0; j < height; j ++)
          for (int i = 0; i < width; i ++) {
            const short gray = (short)*(pixmap ++);
            image(j, i, 0) = gray;
            image(j, i, 1) = gray;
            image(j, i, 2) = gray;
          }
      }

      else { // RGB pixmap
        for (int j = 0; j < height; j ++)
          for (int i = 0; i < width; i ++) {
            image(j, i, 0) = (short)*(pixmap ++);
            image(j, i, 1) = (short)*(pixmap ++);
            image(j, i, 2) = (short)*(pixmap ++);
          }
      }
    }

  }

  // Fills a pixmap from an image object
  void Image::fillPixmap(unsigned char* pixmap, int n_planes_pixmap, const Image& image)
  {
    const int width = image.getWidth();
    const int height = image.getHeight();

    // Grayscale image
    if (image.getNPlanes() == 1)
    {
      // Grayscale pixmap
      if (n_planes_pixmap == 1)
      {
        for (int j = 0; j < height; j ++)
          for (int i = 0; i < width; i ++)
          {
            *(pixmap ++) = image(j, i, 0);
          }
      }

      // RGB pixmap
      else
      {
        for (int j = 0; j < height; j ++)
          for (int i = 0; i < width; i ++)
          {
            *(pixmap ++) = image(j, i, 0);
            *(pixmap ++) = image(j, i, 0);
            *(pixmap ++) = image(j, i, 0);
          }
      }
    }

    // RGB image
    else if (image.getNPlanes() == 3)
    {
      // Grayscale pixmap
      if (n_planes_pixmap == 1)
      {
        for (int j = 0; j < height; j ++)
          for (int i = 0; i < width; i ++)
          {
            *(pixmap ++) = rgb_to_gray(
                (unsigned char)image(j, i, 0),
                (unsigned char)image(j, i, 1),
                (unsigned char)image(j, i, 2));
          }
      }

      // RGB pixmap
      else
      {
        for (int j = 0; j < height; j ++)
          for (int i = 0; i < width; i ++)
          {
            *(pixmap ++) = image(j, i, 0);
            *(pixmap ++) = image(j, i, 1);
            *(pixmap ++) = image(j, i, 2);
          }
      }
    }
  }

  // Draw a pixel in the image
  void Image::drawPixel(int x, int y, const Color& color)
  {
    (*m_setPixelCallback)(this, x, y, color);
  }

  // Various functions for changing some pixel for 1D/3D images
  void Image::setPixel1DChar(ShortTensor* data, int x, int y, const Color& color)
  {
    (*data)(y, x, 0) = color.data0;
  }

  void Image::setPixel3DChar(ShortTensor* data, int x, int y, const Color& color)
  {
    (*data)(y, x, 0) = color.data0;
    (*data)(y, x, 1) = color.data1;
    (*data)(y, x, 2) = color.data2;
  }

  // Draw a cross centered in the given point with the given radius
  void Image::drawCross(int x, int y, int r, const Color& color)
  {
    drawLine(x - r, y, x + r, y, color);
    drawLine(x, y - r, x, y + r, color);
  }

  // Draw a line P1-P2 in the image
  void Image::drawLine(int x1, int y1, int x2, int y2, const Color& color)
  {
    // THE EXTREMELY FAST LINE ALGORITHM Variation E (Addition Fixed Point PreCalc)
    // See original code in FastLine.cc

    int x = x1;
    int y = y1;

    bool yLonger=false;
    int shortLen=y2-y;
    int longLen=x2-x;
    if (abs(shortLen)>abs(longLen))
    {
      int swap=shortLen;
      shortLen=longLen;
      longLen=swap;
      yLonger=true;
    }
    int decInc;
    if (longLen==0) decInc=0;
    else decInc = (shortLen << 16) / longLen;

    if (yLonger)
    {
      if (longLen>0)
      {
        longLen+=y;
        for (int j=0x8000+(x<<16);y<=longLen;++y)
        {
          drawPixel(j >> 16,y, color);
          j+=decInc;
        }
        return;
      }
      longLen+=y;
      for (int j=0x8000+(x<<16);y>=longLen;--y)
      {
        drawPixel(j >> 16,y,color);
        j-=decInc;
      }
      return;
    }

    if (longLen>0)
    {
      longLen+=x;
      for (int j=0x8000+(y<<16);x<=longLen;++x)
      {
        drawPixel(x,j >> 16,color);
        j+=decInc;
      }
      return;
    }
    longLen+=x;
    for (int j=0x8000+(y<<16);x>=longLen;--x)
    {
      drawPixel(x,j >> 16,color);
      j-=decInc;
    }
  }

  // Draw a rectangle in the image.
  void Image::drawRect(int x, int y, int w, int h, const Color& color)
  {
    if (x >= 0 && y >= 0 && x + w < getWidth() && y + h < getHeight())
    {
      drawLine(x, y, x + w, y, color);
      drawLine(x + w, y, x + w, y + h, color);
      drawLine(x + w, y + h, x, y + h, color);
      drawLine(x, y + h, x, y, color);
    }
  }

  void Image::drawRect(const sRect2D& rect, const Color& color)
  {
    return drawRect(rect.x, rect.y, rect.w, rect.h, color);
  }

  // Access functions
  int Image::getWidth() const
  {
    return size(1);
  }

  int Image::getHeight() const
  {
    return size(0);
  }

  int Image::getNPlanes() const
  {
    return size(2);
  }

}
