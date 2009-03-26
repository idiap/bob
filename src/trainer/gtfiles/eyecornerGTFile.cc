#include "eyecornerGTFile.h"

namespace Torch {

eyecornerGTFile::eyecornerGTFile() : GTFile(6)
{
}

bool eyecornerGTFile::load(File* file)
{
	const bool verbose = getBOption("verbose");

	if(verbose) message("eyecornerGTFile::load() ...");

	// IDIAP format (eyes corner)
	float l_eye_l_x, l_eye_l_y;
	file->scanf("%f", &l_eye_l_x);
	file->scanf("%f", &l_eye_l_y);
	float l_eye_r_x, l_eye_r_y;
	file->scanf("%f", &l_eye_r_x);
	file->scanf("%f", &l_eye_r_y);

	m_points[0].x = l_eye_l_x;
	m_points[0].y = l_eye_l_y;
	m_points[1].x = l_eye_r_x;
	m_points[1].y = l_eye_r_y;
	m_points[4].x = (l_eye_l_x + l_eye_r_x) / 2.0;
	m_points[4].y = (l_eye_l_y + l_eye_r_y) / 2.0;

	float r_eye_l_x, r_eye_l_y;
	file->scanf("%f", &r_eye_l_x);
	file->scanf("%f", &r_eye_l_y);
	float r_eye_r_x, r_eye_r_y;
	file->scanf("%f", &r_eye_r_x);
	file->scanf("%f", &r_eye_r_y);

	m_points[2].x = r_eye_l_x;
	m_points[2].y = r_eye_l_y;
	m_points[3].x = r_eye_r_x;
	m_points[3].y = r_eye_r_y;
	m_points[5].x = (r_eye_l_x + r_eye_r_x) / 2.0;
	m_points[5].y = (r_eye_l_y + r_eye_r_y) / 2.0;

	if(verbose)
	{
		message("%s", getName());
		for(int i = 0 ; i < 6 ; i++)
			print(" [%d] %s (%.1f-%.1f)\n", i, getLabel(i), getPoints()[i].x, getPoints()[i].y);
	}

	return true;
}

const char* eyecornerGTFile::getLabel(int i)
{
   	if(i < 0 || i >= m_n_points) return "Out of range label index";
   	switch(i)
	{
	case 0: return "Outer Left Eye Corner";
	case 1: return "Inner Left Eye Corner";
	case 2: return "Outer Right Eye Corner";
	case 3: return "Inner Right Eye Corner";
	case 4: return "Left Eye Center";
	case 5: return "Right Eye Center";
	default: return "Undefined Label";
	}
}

const char* eyecornerGTFile::getName()
{
	return "Eye Corners + computed Eye Centers";
}

eyecornerGTFile::~eyecornerGTFile()
{
}

}



