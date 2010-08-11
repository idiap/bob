#include "ip/pgmImageFile.h"
#include "ip/Image.h"

namespace Torch {

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Constructor

pgmImageFile::pgmImageFile()
	:	ImageFile()
{
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Destructor

pgmImageFile::~pgmImageFile()
{
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Read the image header

bool pgmImageFile::readHeader(Image& image)
{
	char buffer[200];

   	// Reads Magic Number
	if(!m_file.gets(buffer, 200))
	{
		Torch::print("pgmImageFile::readHeader - [magic number] not enable to read pgm file\n");
		return false;
	}
	if(strcmp(buffer, "P5\n") != 0)
	{
		Torch::print("pgmImageFile::readHeader - [magic number] incorrect magic number in file\n");
		return false;
	}

	// Reads comments
	if(!m_file.gets(buffer, 200))
	{
		Torch::print("pgmImageFile::readHeader - [comments] not enable to read pgm file\n");
		return false;
	}
	//while(buffer[0] == '#')
	while(buffer[0] == '#' || buffer[0] == '\n')
		if(!m_file.gets(buffer, 200))
		{
			Torch::print("pgmImageFile::readHeader - [comments] not enable to read pgm in file\n");
			return false;
		}

	//
	// Reads header
	int width = 0;
	int height = 0;

   	sscanf(buffer, "%d %d\n", &width, &height);
	if(height == 0)
	{
		if(!m_file.gets(buffer, 200))
		{
			Torch::print("pgmImageFile::readHeader - [image size] not enable to read pgm file\n");
			return false;
		}
   		sscanf(buffer, "%d\n", &height);
	}

   	if(width <= 0)
	{
		Torch::print("pgmImageFile::readHeader - incorrect width (%d) in pgm file\n", width);
		return false;
	}
   	if(height <= 0)
	{
		Torch::print("pgmImageFile::readHeader - incorrect height (%d) in pgm file\n", height);
		return false;
	}

	//
	// Reads resolution
	if(!m_file.gets(buffer, 200))
	{
		Torch::print("pgmImageFile::readHeader - [depth] not enable to read pgm file\n");
		return false;
	}
	if(strcmp(buffer, "255\n") != 0)
	{
	   	Torch::print("pgmImageFile::readHeader - incorrect resolution (%s) in pgm file, only 255 is supported\n", buffer);
		return false;
	}

	// OK, resize the image to the new dimensions
	return image.resize(width, height, image.getNPlanes());// Keep the number of color channels
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Read the image pixmap

bool pgmImageFile::readPixmap(Image& image)
{
	const int width = image.getWidth();
	const int height = image.getHeight();
	const int n_bytes = 1 * width * height;

	// Read the pixmap
	unsigned char* pixmap = new unsigned char[n_bytes];
	if (m_file.read(pixmap, sizeof(unsigned char), n_bytes) != n_bytes)
	{
		delete[] pixmap;
		return false;
	}

	// OK, update the image pixels to the pixmap
	Image::fillImage(pixmap, 1, image);
	delete[] pixmap;
	return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Write the image header

bool pgmImageFile::writeHeader(const Image& image)
{
	m_file.printf("P5\n");
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

bool pgmImageFile::writePixmap(const Image& image)
{
	const int width = image.getWidth();
	const int height = image.getHeight();
	const int n_bytes = 1 * width * height;

	// Fill the pixmap with the image pixels
	unsigned char* pixmap = new unsigned char[n_bytes];
	Image::fillPixmap(pixmap, 1, image);

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
