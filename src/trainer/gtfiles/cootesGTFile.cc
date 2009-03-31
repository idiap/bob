#include "cootesGTFile.h"

namespace Torch {

cootesGTFile::cootesGTFile() : GTFile(68)
{
	CHECK_FATAL(setLabel(31, "leye_center") == true);
	CHECK_FATAL(setLabel(36, "reye_center") == true);
}

bool cootesGTFile::load(File* file)
{
	const bool verbose = getBOption("verbose");

	if(verbose) message("cootesGTFile::load() ...");

	char buffer[250];
	do
	{
		file->gets(buffer, 250);
	} while(buffer[0] != '{');

	for(int i = 0 ; i < 68 ; i++)
	{
	   	float x, y;
		file->scanf("%g", &x);
		file->scanf("%g", &y);
		m_points[i].x = x;
		m_points[i].y = y;
	}

	if(verbose)
	{
		message("%s", getName());
		for(int i = 0 ; i < 68 ; i++)
			print(" [%d] %s (%.1f-%.1f)\n", i, getLabel(i), m_points[i].x, m_points[i].y);
	}

	return true;
}

const char* cootesGTFile::getName()
{
	return "Tim Cootes format";
}

cootesGTFile::~cootesGTFile()
{
}

}



