#include "pgmImageFile.h"
#include "Image.h"

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
	if(!gets(buffer, 200))
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
	if(!gets(buffer, 200))
	{
		Torch::print("pgmImageFile::readHeader - [comments] not enable to read pgm file\n");
		return false;
	}
	//while(buffer[0] == '#')
	while(buffer[0] == '#' || buffer[0] == '\n')
		if(!gets(buffer, 200))
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
		if(!gets(buffer, 200))
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
	if(!gets(buffer, 200))
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
	if (read(pixmap, sizeof(unsigned char), n_bytes) != n_bytes)
	{
		delete[] pixmap;
		return false;
	}

	// OK, update the image pixels to the pixmap
	ImageFile::fillImage(pixmap, 1, image);
	delete[] pixmap;
	return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Write the image header

bool pgmImageFile::writeHeader(const Image& image)
{
	printf("P5\n");
	printf("#\n");
	printf("# Image generated using Torch vision\n");
	printf("# (c) 2004-2008 Sebastien Marcel marcel@idiap.ch\n");
	printf("# IDIAP Research Institute\n");
	printf("#\n");
	printf("%d %d\n", image.getWidth(), image.getHeight());
	printf("255\n");

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
	ImageFile::fillPixmap(pixmap, 1, image);

	// Write the pixmap
	if (write(pixmap, sizeof(unsigned char), n_bytes) != n_bytes)
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
