#include "scanning/eyecenterGTFile.h"

namespace Torch {

eyecenterGTFile::eyecenterGTFile() : GTFile(2)
{
	CHECK_FATAL(setLabel(0, "leye_center") == true);
	CHECK_FATAL(setLabel(1, "reye_center") == true);
}

bool eyecenterGTFile::load(File* file)
{
	const bool verbose = getBOption("verbose");

	if(verbose) message("eyecenterGTFile::load() ...");

	// IDIAP format (eyes center)
	float x, y;

	// left eye
	file->scanf("%f", &x);
	file->scanf("%f", &y);

	m_points[0].x = x;
	m_points[0].y = y;

	// right eye
	file->scanf("%f", &x);
	file->scanf("%f", &y);

	m_points[1].x = x;
	m_points[1].y = y;


	if(verbose)
	{
		message("%s", getName());
		for(int i = 0 ; i < m_n_points ; i++)
			print(" [%d] %s (%.1f-%.1f)\n", i, getLabel(i), m_points[i].x, m_points[i].y);
	}

	return true;
}

const char* eyecenterGTFile::getName()
{
	return "Idiap Eye Centers";
}

eyecenterGTFile::~eyecenterGTFile()
{
}

}
