#include "trainer/bancaGTFile.h"

namespace Torch {

bancaGTFile::bancaGTFile() : GTFile(10)
{
	CHECK_FATAL(setLabel(1, "leye_center") == true);
	CHECK_FATAL(setLabel(4, "reye_center") == true);
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
			print(" [%d] %s (%.1f-%.1f)\n", i, getLabel(i), m_points[i].x, m_points[i].y);
	}

	return true;
}

const char* bancaGTFile::getName()
{
	return "BANCA format";
}

bancaGTFile::~bancaGTFile()
{
}

}



