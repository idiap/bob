#include "frontalEyeNoseChinGTFile.h"

namespace Torch {

frontalEyeNoseChinGTFile::frontalEyeNoseChinGTFile() : GTFile(8)
{
}

bool frontalEyeNoseChinGTFile::load(File* file)
{
	const bool verbose = getBOption("verbose");

	if(verbose) message("frontalEyeNoseChinGTFile::load() ...");

	int n_;

	file->scanf("%d", &n_);

	if(n_ != 6) error("Number of points (%d) <> 6", n_);

	// left eye
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
	m_points[6].x = (l_eye_l_x + l_eye_r_x) / 2.0;
	m_points[6].y = (l_eye_l_y + l_eye_r_y) / 2.0;

	// right eye
	float r_eye_l_x, r_eye_l_y;
	float r_eye_r_x, r_eye_r_y;
	file->scanf("%f", &r_eye_r_x);
	file->scanf("%f", &r_eye_r_y);
	file->scanf("%f", &r_eye_l_x);
	file->scanf("%f", &r_eye_l_y);

	m_points[2].x = r_eye_r_x;
	m_points[2].y = r_eye_r_y;
	m_points[3].x = r_eye_l_x;
	m_points[3].y = r_eye_l_y;
	m_points[7].x = (r_eye_l_x + r_eye_r_x) / 2.0;
	m_points[7].y = (r_eye_l_y + r_eye_r_y) / 2.0;

	// nose tip
	float x, y;
	file->scanf("%f", &x);
	file->scanf("%f", &y);

	m_points[4].x = x;
	m_points[4].y = y;

	// chin
	file->scanf("%f", &x);
	file->scanf("%f", &y);

	m_points[5].x = x;
	m_points[5].y = y;

	if(verbose)
	{
		message("%s", getName());
		for(int i = 0 ; i < m_n_points ; i++)
			print(" [%d] %s (%.1f-%.1f)\n", i, getLabel(i), getPoints()[i].x, getPoints()[i].y);
	}

	return true;
}

char *frontalEyeNoseChinGTFile::getLabel(int i)
{
   	if(i < 0 || i >= m_n_points) return "Out of range label index";
   	switch(i)
	{
	case 0: return "Outer Left Eye Corner";
	case 1: return "Inner Left Eye Corner";
	case 2: return "Inner Right Eye Corner";
	case 3: return "Outer Right Eye Corner";
	case 4: return "Nose Tip";
	case 5: return "Chin";
	case 6: return "Left Eye Center";
	case 7: return "Right Eye Center";
	default: return "Undefined Label";
	}
}

char *frontalEyeNoseChinGTFile::getName()
{
	return "Eye Corners + computed Eye Centers + Nose tip + Chin";
}

frontalEyeNoseChinGTFile::~frontalEyeNoseChinGTFile()
{
}

}

