#include "cootesGTFile.h"

namespace Torch {

cootesGTFile::cootesGTFile() : GTFile(68)
{
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
			print(" [%d] %s (%.1f-%.1f)\n", i, getLabel(i), getPoints()[i].x, getPoints()[i].y);
	}

	return true;
}

char *cootesGTFile::getLabel(int i)
{
   	if(i < 0 || i >= m_n_points) return "Out of range label index";
   	switch(i)
	{
	case 31: return "Left Eye Center";
	case 36: return "Right Eye Center";
	default: return "Undefined Label";
	}
}

char *cootesGTFile::getName()
{
	return "Tim Cootes format";
}

cootesGTFile::~cootesGTFile()
{
}

}



