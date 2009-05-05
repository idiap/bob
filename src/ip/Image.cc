#include "Image.h"
#include "Color.h"
#include "vision.h"

namespace Torch {

/////////////////////////////////////////////////////////////////////////////////
// Constructor

Image::Image(int width, int height, int n_planes)
	:	ShortTensor(height, width, n_planes),
		m_setPixelCallback(setPixel1DChar)
{
	resize(width, height, n_planes);
}

/////////////////////////////////////////////////////////////////////////////////
// Destructor

Image::~Image()
{
	cleanup();
}

/////////////////////////////////////////////////////////////////////////////////
// Resize the image (to new dimensions, no of planes and storage type)

bool Image::resize(int width, int height, int n_planes)
{
	// Check parameters
	if (	width < 1 || height < 1 ||
		(n_planes != 1 && n_planes != 3))// Suport only images with one (gray) or 3 (RGB like) components
	{
		Torch::message("Torch::Image::resize - invalid parameters!\n");
		return false;
	}

	// Resize only if needed
	if (getWidth() != width || getHeight() != height)
	{
		cleanup();

		// Allocate the tensor, fill it black and set the callback to change the pixels
		ShortTensor::t = THShortTensor_newWithSize3d(height, width, n_planes);
		m_setPixelCallback = n_planes == 1 ? setPixel1DChar : setPixel3DChar;
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////////////
// Copy from some 3D tensor (should have the same dimension)
//	- all planes are copied

bool Image::copyFrom(const Tensor& data)
{
	// Check parameters
	if (	ShortTensor::t == 0 ||
		data.nDimension() != nDimension())
	{
		Torch::message("Torch::Image::copyFrom - invalid parameters!\n");
		return false;
	}
	for (int i = 0; i < data.nDimension(); i ++)
		if (data.size(i) != size(i))
		{
			Torch::message("Torch::Image::copyFrom - invalid parameters!\n");
			return false;
		}

	// OK, copy it
	ShortTensor::copy(&data);
	return true;
}

/////////////////////////////////////////////////////////////////////////////////
// Copy from another image (should have the same dimension)
//	- all planes are copied

bool Image::copyFrom(const Image& image)
{
        // Check parameters
	if (    ShortTensor::t == 0 || image.t == 0 ||
                getWidth() != image.getWidth() ||
                getHeight() != image.getHeight())
	{
	        Torch::message("Torch::Image::copyFrom - invalid image!\n");
		return false;
	}

        // Copy the image
        const int w = getWidth();
        const int h = getHeight();
        if (getNPlanes() == image.getNPlanes())
        {
                // The same number of planes!
                ShortTensor::copy(&image);
        }
        else if (getNPlanes() == 1)
        {
                // RGB to gray
                for (int y = 0; y < h; y ++)
                        for (int x = 0; x < w; x ++)
                        {
                                set(y, x, 0, rgb_to_gray(
                                        image.get(y, x, 0),
                                        image.get(y, x, 1),
                                        image.get(y, x, 2)));
                        }
        }
        else if (getNPlanes() == 3)
        {
                // gray to RGB
                for (int y = 0; y < h; y ++)
                        for (int x = 0; x < w; x ++)
                        {
                                const short gray = image.get(y, x, 0);
                                set(y, x, 0, gray);
                                set(y, x, 1, gray);
                                set(y, x, 2, gray);
                        }
        }

        // OK
        return true;
}

/////////////////////////////////////////////////////////////////////////////////
// Draw a pixel in the image

void Image::drawPixel(int x, int y, const Color& color)
{
	(*m_setPixelCallback)(this, x, y, color);
}

/////////////////////////////////////////////////////////////////////////////////
// Various functions for changing some pixel for 1D/3D images

void Image::setPixel1DChar(ShortTensor* data, int x, int y, const Color& color)
{
	data->set(y, x, 0, color.data0);
}

void Image::setPixel3DChar(ShortTensor* data, int x, int y, const Color& color)
{
	data->set(y, x, 0, color.data0);
	data->set(y, x, 1, color.data1);
	data->set(y, x, 2, color.data2);
}

/////////////////////////////////////////////////////////////////////////////////
// Draw a cross centered in the given point with the given radius

void Image::drawCross(int x, int y, int r, const Color& color)
{
	drawLine(x - r, y, x + r, y, green);
	drawLine(x, y - r, x, y + r, green);
}

/////////////////////////////////////////////////////////////////////////////////
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

/////////////////////////////////////////////////////////////////////////////////////////
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

/////////////////////////////////////////////////////////////////////////////////////////
// Access functions

int Image::getWidth() const
{
	return ShortTensor::t == 0 ? -1 : size(1);
}

int Image::getHeight() const
{
	return ShortTensor::t == 0 ? -1 : size(0);
}

int Image::getNPlanes() const
{
	return ShortTensor::t == 0 ? -1 : size(2);
}

/////////////////////////////////////////////////////////////////////////////////////
// Delete the allocate memory

void Image::cleanup()
{
	THShortTensor_free(ShortTensor::t);
	ShortTensor::t = 0;
}

/////////////////////////////////////////////////////////////////////////////////

}
