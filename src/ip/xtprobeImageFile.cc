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

ImageFile* GetImageFile(const char* filename)
{
	const int i = FindImageFormat(filename);
	if (i != -1)
	{
        	switch(i)
		{
		case 0:
			return new ppmImageFile();

		case 1:
		case 2:
		case 3:
		case 4:
			return new jpegImageFile();

		case 5:
		case 6:
		   	return new gifImageFile();

		case 7:
		   	return new pgmImageFile();

		case 8:
		case 9:
		case 10:
		case 11:
		   	return new tiffImageFile();
			break;

		default:
			warning("xtprobeImageFile::open (%s) Impossible to probe image format from extension.",
				filename);
			return 0;
		}
	}
	else
	{
		return 0;
	}
}

//////////////////////////////////////////////////////////////////////////////////////
// Constructor

xtprobeImageFile::xtprobeImageFile()
	:	m_loader(0)
{
}

//////////////////////////////////////////////////////////////////////////////////////
// Destructor

xtprobeImageFile::~xtprobeImageFile()
{
	if(m_loader != 0) delete m_loader;
}

//////////////////////////////////////////////////////////////////////////////////////
// Save an image

bool xtprobeImageFile::save(const Image& image, const char* filename)
{
	delete m_loader;
	m_loader = GetImageFile(filename);
	if (m_loader == 0)
	{
	   	warning("xtprobeImageFile::save(): failed to allocate image loader.");
		return false;
	}
	return m_loader->save(image, filename);
}

//////////////////////////////////////////////////////////////////////////////////////
// Load an image

bool xtprobeImageFile::load(Image& image, const char* filename)
{
	delete m_loader;
	m_loader = GetImageFile(filename);
	if (m_loader == 0)
	{
	   	warning("xtprobeImageFile::load(): failed to allocate image loader.");
		return false;
	}
	return m_loader->load(image, filename);
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
