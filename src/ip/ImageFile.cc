#include "ip/ImageFile.h"
#include "ip/Image.h"
#include "ip/Convert.h"

namespace Torch {

///////////////////////////////////////////////////////////////////////////////
// Constructor

ImageFile::ImageFile()
{
}

///////////////////////////////////////////////////////////////////////////////
// Destructor

ImageFile::~ImageFile()
{
	m_file.close();
}

///////////////////////////////////////////////////////////////////////////////
// Save an image

bool ImageFile::save(const Image& image, const char* filename)
{
	bool ret = false;

	if (m_file.open(filename, "w") && writeHeader(image) && writePixmap(image))
	{
		ret = true;
	}
	m_file.close();

	return ret;
}

///////////////////////////////////////////////////////////////////////////////
// Load an image

bool ImageFile::load(Image& image, const char* filename)
{
	bool ret = false;

	if (m_file.open(filename, "r") && readHeader(image) && readPixmap(image))
	{
		ret = true;
	}
	m_file.close();

	return ret;
}

}
