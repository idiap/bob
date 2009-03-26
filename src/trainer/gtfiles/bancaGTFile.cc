#include "bancaGTFile.h"

namespace Torch {

bancaGTFile::bancaGTFile() : GTFile(10)
{
}

bool bancaGTFile::load(File* file)
{
	const bool verbose = getBOption("verbose");

	if(verbose) message("bancaGTFile::load() ...");

	// BANCA format
	float x, y;

	for(int i = 0 ; i < 10 ; i++)
	{
		file->scanf("%f", &x);
		file->scanf("%f", &y);
		m_points[i].x = x;
		m_points[i].y = y;
	}

	if(verbose)
	{
		message("%s", getName());
		for(int i = 0 ; i < m_n_points ; i++)
			print(" [%d] %s (%.1f-%.1f)\n", i, getLabel(i), getPoints()[i].x, getPoints()[i].y);
	}

	return true;
}

const char* bancaGTFile::getLabel(int i)
{
   	if(i < 0 || i >= m_n_points) return "Out of range label index";
   	switch(i)
	{
	case 0: return "Undefined Label";
	case 1: return "Left Eye Center";
	case 2: return "Undefined Label";
	case 3: return "Undefined Label";
	case 4: return "Right Eye Center";
	case 5: return "Undefined Label";
	case 6: return "Undefined Label";
	case 7: return "Undefined Label";
	case 8: return "Undefined Label";
	case 9: return "Undefined Label";
	default: return "Undefined Label";
	}
}

const char* bancaGTFile::getName()
{
	return "BANCA format";
}

bancaGTFile::~bancaGTFile()
{
}

}



