#include "scanning/profileEyeNoseChinGTFile.h"

namespace Torch {

profileEyeNoseChinGTFile::profileEyeNoseChinGTFile() : GTFile(3)
{
	CHECK_FATAL(setLabel(0, "leye_center") == true);
	CHECK_FATAL(setLabel(1, "nose_tip") == true);
	CHECK_FATAL(setLabel(2, "chin") == true);
}

bool profileEyeNoseChinGTFile::load(File* file)
{
	const bool verbose = getBOption("verbose");

	if(verbose) message("profileEyeNoseChinGTFile::load() ...");

	int n_;

	file->scanf("%d", &n_);

	if(n_ != 3) error("Number of points (%d) <> 3", n_);

	// left eye
	float x, y;
	file->scanf("%f", &x);
	file->scanf("%f", &y);

	m_points[0].x = x;
	m_points[0].y = y;

	// nose tip
	file->scanf("%f", &x);
	file->scanf("%f", &y);

	m_points[1].x = x;
	m_points[1].y = y;

	// chin
	file->scanf("%f", &x);
	file->scanf("%f", &y);

	m_points[2].x = x;
	m_points[2].y = y;

	if(verbose)
	{
		message("%s", getName());
		for(int i = 0 ; i < m_n_points ; i++)
			print(" [%d] %s (%.1f-%.1f)\n", i, getLabel(i), m_points[i].x, m_points[i].y);
	}

	return true;
}

const char* profileEyeNoseChinGTFile::getName()
{
	return "Left Eye Center + Nose Tip + Chin (profile)";
}

profileEyeNoseChinGTFile::~profileEyeNoseChinGTFile()
{
}

}

