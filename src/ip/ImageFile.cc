#include "ImageFile.h"
#include "Image.h"
#include "Convert.h"

namespace Torch {

///////////////////////////////////////////////////////////////////////////////
// Constructor

ImageFile::ImageFile()
	:	File()
{
}

///////////////////////////////////////////////////////////////////////////////
// Destructor

ImageFile::~ImageFile()
{
}

///////////////////////////////////////////////////////////////////////////////
// Fills a pixmap from an image object

void ImageFile::fillPixmap(unsigned char* pixmap, int n_planes_pixmap, const Image& image)
{
	const int width = image.getWidth();
	const int height = image.getHeight();

	unsigned char* ppixmap = pixmap;

	// Grayscale image
	if (image.getNPlanes() == 1)
	{
		// Grayscale pixmap
		if (n_planes_pixmap == 1)
		{
			for (int j = 0; j < height; j ++)
				for (int i = 0; i < width; i ++)
				{
					*(pixmap ++) = image.get(j, i, 0);
				}
		}

		// RGB pixmap
		else
		{
			for (int j = 0; j < height; j ++)
				for (int i = 0; i < width; i ++)
				{
					*(pixmap ++) = image.get(j, i, 0);
					*(pixmap ++) = image.get(j, i, 0);
					*(pixmap ++) = image.get(j, i, 0);
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
							(unsigned char)image.get(j, i, 0),
							(unsigned char)image.get(j, i, 1),
							(unsigned char)image.get(j, i, 2));
				}
		}

		// RGB pixmap
		else
		{
			for (int j = 0; j < height; j ++)
				for (int i = 0; i < width; i ++)
				{
					*(pixmap ++) = image.get(j, i, 0);
					*(pixmap ++) = image.get(j, i, 1);
					*(pixmap ++) = image.get(j, i, 2);
				}
		}
	}
}

///////////////////////////////////////////////////////////////////////////////
// Fills an image object from a pixmap

void ImageFile::fillImage(const unsigned char* pixmap, int n_planes_pixmap, Image& image)
{
	const int width = image.getWidth();
	const int height = image.getHeight();

	const unsigned char* ppixmap = pixmap;

	// Grayscale image
	if (image.getNPlanes() == 1)
	{
		// Grayscale pixmap
		if (n_planes_pixmap == 1)
		{
			for (int j = 0; j < height; j ++)
				for (int i = 0; i < width; i ++)
				{
					image.set(j, i, 0, (short)*(pixmap ++));
				}
		}

		// RGB pixmap
		else
		{
			for (int j = 0; j < height; j ++)
				for (int i = 0; i < width; i ++)
				{
					image.set(j, i, 0, (short)
						rgb_to_gray(*(pixmap ++), *(pixmap ++), *(pixmap ++)));
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
					const short gray = (short)*(pixmap ++);
					image.set(j, i, 0, gray);
					image.set(j, i, 1, gray);
					image.set(j, i, 2, gray);
				}
		}

		// RGB pixmap
		else
		{
			for (int j = 0; j < height; j ++)
				for (int i = 0; i < width; i ++)
				{
					image.set(j, i, 0, (short)*(pixmap ++));
					image.set(j, i, 1, (short)*(pixmap ++));
					image.set(j, i, 2, (short)*(pixmap ++));
				}
		}
	}
}

///////////////////////////////////////////////////////////////////////////////

}
