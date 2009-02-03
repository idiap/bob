#include "xtprobeImageFile.h"

#include "pgmImageFile.h"
#include "ppmImageFile.h"
#include "jpegImageFile.h"
#include "gifImageFile.h"
#include "tiffImageFile.h"

namespace Torch {

//////////////////////////////////////////////////////////////////////////////////////
// Supported image formats

static int n_imageformat = 12;

struct imageformat_
{
   	const char *extension;
	const char *coding;
	int n_planes;
};

static struct imageformat_ imageformat[] = {	{"ppm", "rgb", 3},
	                                	{"jpg", "rgb", 3},
                	                	{"jpeg", "rgb", 3},
                        	        	{"JPG", "rgb", 3},
                                		{"JPEG", "rgb", 3},
	                                	{"gif", "rgb", 3},
        	                        	{"GIF", "rgb", 3},
                	                	{"pgm", "gray", 1},
                	                	{"TIF", "rgb", 3},
                	                	{"TIFF", "rgb", 3},
                	                	{"tif", "rgb", 3},
                	                	{"tiff", "rgb", 3}};

int FindImageFormat(const char *filename);

//////////////////////////////////////////////////////////////////////////////////////
// Constructor

xtprobeImageFile::xtprobeImageFile()
	:	ImageFile(),
		m_loader(0)
{
}

//////////////////////////////////////////////////////////////////////////////////////
// Destructor

xtprobeImageFile::~xtprobeImageFile()
{
	delete m_loader;
}

//////////////////////////////////////////////////////////////////////////////////////
// Read the image header

bool xtprobeImageFile::readHeader(Image& image)
{
	return 	m_loader != 0 &&
		m_loader->isOpened() &&
		m_loader->readHeader(image);
}

//////////////////////////////////////////////////////////////////////////////////////
// Read the image pixmap

bool xtprobeImageFile::readPixmap(Image& image)
{
	return 	m_loader != 0 &&
		m_loader->isOpened() &&
		m_loader->readPixmap(image);
}

//////////////////////////////////////////////////////////////////////////////////////
// Write the image header

bool xtprobeImageFile::writeHeader(const Image& image)
{
	return	m_loader != 0 &&
		m_loader->isOpened() &&
		m_loader->writeHeader(image);
}

//////////////////////////////////////////////////////////////////////////////////////
// Write the image pixmap

bool xtprobeImageFile::writePixmap(const Image& image)
{
	return 	m_loader != 0 &&
		m_loader->isOpened() &&
		m_loader->writePixmap(image);
}

//////////////////////////////////////////////////////////////////////////////////////
// Loads some file and check for its image format - overriden

bool xtprobeImageFile::open(const char* file_name, const char* open_flags)
{
	File::close();
	delete m_loader;
	m_loader = 0;

	// Check the image format
	const int i = FindImageFormat(file_name);
	if (i != -1)
	{
        	//const char* coding = imageformat[i].coding;
		//const int n_planes = imageformat[i].n_planes;

		switch(i)
		{
		case 0:
			m_loader = new ppmImageFile();
		   	break;
		case 1:
		case 2:
		case 3:
		case 4:
			m_loader = new jpegImageFile();
			break;
		case 5:
		case 6:
		   	m_loader = new gifImageFile();
			break;
		case 7:
		   	m_loader = new pgmImageFile();
			break;
		case 8:
		case 9:
		case 10:
		case 11:
		   	m_loader = new tiffImageFile();
			break;

		default:
			warning("xtprobeImageFile::open (%s) Impossible to probe image format from extension.",
				file_name);
			return false;
		}
	}
	else
	{
		return false;
	}

	// OK, try to open the file with the choosen loader
	return m_loader->open(file_name, open_flags);
}

//////////////////////////////////////////////////////////////////////////////////////
// Find an image format using the extension of the given filename

int FindImageFormat(const char *filename)
{
	bool verbose = false;
	char *extension;
	char *str = (char *) filename;

	/*
	if(str[0] == '.')
	{
		str++;
		if(str[0] == '.') str ++;
	}
	*/

	//extension = (char *) rindex(filename, '.');
	extension = strrchr(str, '.');
	if (extension == NULL)
	{
		return -1;
	}
	extension++;

	if(verbose == true)
	{
		print("Checking file format of %s (%s):\n", filename, extension);
		print(" + Trying");
	}

 	for(int i = 0 ; i < n_imageformat ; i++)
        {
        	if(verbose == true) print(" %s", imageformat[i].extension);

                if(strcmp(extension, imageformat[i].extension) == 0)
                {
                	if(verbose == true) print(" [OK]\n");

                        return i;
                } else if(verbose == true) print("...");
        }

	if(verbose == true) print("\n");

	return -1;
}

//////////////////////////////////////////////////////////////////////////////////////

}
