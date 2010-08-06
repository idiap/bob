#include "bbx2eye19x19deye10GTFile.h"

/**
 * Converts a bounding box into eye positions
 *
 * @param x The top left ordinate of the bounding box
 * @param y The top left abcissa of the bounding box
 * @param w The width of the bounding box
 * @param leye The estimated position that will be set for the left eye
 * @param reye The estimated position that will be set for the right eye
 */
static void bbx2eyepos(short x, short y, short w,
    Torch::sPoint2D& leye, Torch::sPoint2D& reye)
{
	// use the "hard coded method" for getting the normal idiap eye centers
	// TAKEN FROM MOBIO PROJECT

	const float D_EYES    = 10.0;
	const float Y_UPPER   = 5.0;
	const float bbx_width = 19.0;

	float ratio = (float)w/bbx_width;

	float Rx = ratio * (D_EYES + bbx_width) / 2 + x;
	float Lx = Rx - D_EYES * ratio;

	float Ry = y + ratio * Y_UPPER;
	float Ly = Ry;

	leye.x = Lx + 0.5;
	leye.y = Ly + 0.5;
	reye.x = Rx + 0.5;
	reye.y = Ry + 0.5;
}

namespace Torch {

bbx2eye19x19deye10_GTFile::bbx2eye19x19deye10_GTFile() : GTFile(2)
{
	CHECK_FATAL(setLabel(0, "leye_center") == true);
	CHECK_FATAL(setLabel(1, "reye_center") == true);
}

bool bbx2eye19x19deye10_GTFile::load(const Torch::Pattern& p)
{
  bbx2eyepos(p.m_x, p.m_y, p.m_w, m_points[0], m_points[1]);
  return true;
}

bool bbx2eye19x19deye10_GTFile::load(File* file)
{
	const bool verbose = getBOption("verbose");

	if(verbose) message("bbx2eye19x19deye10_GTFile::load() ...");

	// read up the four ints of the box
	int BBX_x, BBX_y, BBX_width, hh;
	file->scanf("%d", &BBX_x);
	file->scanf("%d", &BBX_y);
	file->scanf("%d", &BBX_width);
	file->scanf("%d", &hh);

  bbx2eyepos(BBX_x, BBX_y, BBX_width, m_points[0], m_points[1]);

	// debug if verbose
	if(verbose)
	{
		message("%s", getName());
		for(int i = 0 ; i < m_n_points ; i++)
			print(" [%d] %s (%.1f-%.1f)\n", i, getLabel(i), m_points[i].x, m_points[i].y);
	}

	return true;
}

const char* bbx2eye19x19deye10_GTFile::getName()
{
	return "Idiap Eye Centers";
}

bbx2eye19x19deye10_GTFile::~bbx2eye19x19deye10_GTFile()
{
}

}
