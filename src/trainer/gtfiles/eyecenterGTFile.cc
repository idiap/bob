#include "eyecenterGTFile.h"

namespace Torch {

eyecenterGTFile::eyecenterGTFile() : GTFile(2)
{
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
			print(" [%d] %s (%.1f-%.1f)\n", i, getLabel(i), getPoints()[i].x, getPoints()[i].y);
	}

	return true;
}

char *eyecenterGTFile::getLabel(int i)
{
   	if(i < 0 || i >= m_n_points) return "Out of range label index";
	if(i == 0) return "Left Eye Center";
	else if(i == 1) return "Right Eye Center";
}

char *eyecenterGTFile::getName()
{
	return "Idiap Eye Centers";
}

eyecenterGTFile::~eyecenterGTFile()
{
}

}
