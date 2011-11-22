/**
 * @file cxx/old/ip/src/ppmImageFile.cc
 * @date Sat Apr 30 18:41:25 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
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
#include "ip/ppmImageFile.h"
#include "ip/Image.h"

namespace Torch {

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Constructor

ppmImageFile::ppmImageFile()
	:	ImageFile()
{
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Destructor

ppmImageFile::~ppmImageFile()
{
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Read the image header

bool ppmImageFile::readHeader(Image& image)
{
	char buffer[200];

   	// Reads Magic Number
	if(!m_file.gets(buffer, 200))
	{
		Torch::print("ppmImageFile::readHeader - not enable to read ppm file\n");
		return false;
	}

	if(strcmp(buffer, "P6\n") != 0)
	{
		Torch::print("ppmImageFile::readHeader - incorrect magic number\n");
		return false;
	}

	// Reads comments
	if(!m_file.gets(buffer, 200))
	{
		Torch::print("ppmImageFile::readHeader - not enable to read ppm file\n");
		return false;
	}
	//while(buffer[0] == '#')
	while(buffer[0] == '#' || buffer[0] == '\n')
		if(!m_file.gets(buffer, 200))
		{
			Torch::print("ppmImageFile::readHeader - not enable to read ppm file\n");
			return false;
		}

	//
	// Reads header
	int width, height;
   	sscanf(buffer, "%d %d\n", &width, &height);
   	if(width <= 0)
	{
		Torch::print("ppmImageFile::readHeader - incorrect width in ppm file\n");
		return false;
	}
   	if(height <= 0)
	{
		Torch::print("ppmImageFile::readHeader -incorrect height in ppm file\n");
		return false;
	}

	//
	// Reads resolution
	if(!m_file.gets(buffer, 200))
	{
		Torch::print("ppmImageFile::readHeader - not enable to read ppm file\n");
		return false;
	}

	if(strcmp(buffer, "255\n") != 0)
	{
		Torch::print("ppmImageFile::readHeader - incorrect resolution, only 255 is supported\n");
		return false;
	}

	// OK, resize the image to the new dimensions
	return image.resize(width, height, image.getNPlanes());// Keep the number of color channels
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Read the image pixmap

bool ppmImageFile::readPixmap(Image& image)
{
	const int width = image.getWidth();
	const int height = image.getHeight();
	const int n_bytes = 3 * width * height;

	// Read the pixmap
	unsigned char* pixmap = new unsigned char[n_bytes];
	if (m_file.read(pixmap, sizeof(unsigned char), n_bytes) != n_bytes)
	{
		delete[] pixmap;
		return false;
	}

	// OK, update the image pixels to the pixmap
	Image::fillImage(pixmap, 3, image);
	delete[] pixmap;
	return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Write the image header

bool ppmImageFile::writeHeader(const Image& image)
{
	m_file.printf("P6\n");
	m_file.printf("#\n");
	m_file.printf("# Image generated using torch5spro\n");
	m_file.printf("# (c) torch5spro 2010\n");
	m_file.printf("# Sebastien Marcel (marcel@idiap.ch) -- Idiap Research Institute (www.idiap.ch)\n");
	m_file.printf("#\n");
	m_file.printf("%d %d\n", image.getWidth(), image.getHeight());
	m_file.printf("255\n");

	return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Write the image pixmap

bool ppmImageFile::writePixmap(const Image& image)
{
	const int width = image.getWidth();
	const int height = image.getHeight();
	const int n_bytes = 3 * width * height;

	// Fill the pixmap with the image pixels
	unsigned char* pixmap = new unsigned char[n_bytes];
	Image::fillPixmap(pixmap, 3, image);

	// Write the pixmap
	if (m_file.write(pixmap, sizeof(unsigned char), n_bytes) != n_bytes)
	{
		delete[] pixmap;
		return false;
	}

	// OK
	delete[] pixmap;
	return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////

}
