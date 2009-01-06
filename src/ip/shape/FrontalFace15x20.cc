#include "FrontalFace15x20.h"

namespace Torch {

FrontalFace15x20::FrontalFace15x20(int postype_) : FaceModel(postype_)
{
	model_width = 15.0;
	model_height = 20.0;
}

bool FrontalFace15x20::ldm2bbx()
{
	const bool verbose = getBOption("verbose");

	double c0x, c0y, EEx, EEy;

	sPoint2D l_eye = getLeye();
	sPoint2D r_eye = getReye();

	EEx = r_eye.x - l_eye.x;
	EEy = r_eye.y - l_eye.y;

	// The face is non oriented
	if(!IS_NEAR(EEy, 0, 0.001))
	   	warning("FrontalFace15x20: y coordinates of eyes should be equal");

	c0x = l_eye.x + EEx / 2.0;
	c0y = l_eye.y + EEy / 2.0;

	if(verbose)
		print("FrontalFace15x20: c0x=%g c0y=%g\n", c0x, c0y);

	/* The width of the face is given by the constant #zy_zy#
	   and the distance between eyes is equal to 2 times #pupil_se# */
	bbx_width = (int) (ZY_ZY * EEx / (2.0 * PUPIL_SE) + 0.5);
	if(bbx_width % 2 == 0) bbx_width += 1;

	// The height of the face is width*20/15
	bbx_height = (int) ((double) bbx_width * model_height / model_width + 0.5);

	int upperleft_x;
	int upperleft_y;

	/* The upperleft x coordinate of the face is given
	   by the middle of the face minus the half size of the face */
	double x_ = (double) bbx_width / 2.0;
	upperleft_x = (int) (l_eye.x + (r_eye.x - l_eye.x) / 2.0 - x_ + 0.5);

	/* The upperleft y coordinate of the face is given
	   by the constants #en_gn# and #tr_gn# */
	upperleft_y = (int) (c0y - (bbx_height * (TR_GN - EN_GN) / TR_GN));

	if(verbose)
		print("FrontalFace15x20: x=%d y=%d w=%d h=%d\n", upperleft_x, upperleft_y, bbx_width, bbx_height);

	// Filling bbx
	bbx_points[0].x = upperleft_x+1;
	bbx_points[0].y = upperleft_y;

	bbx_points[1].x = upperleft_x + bbx_width;
	bbx_points[1].y = upperleft_y;

	bbx_points[2].x = upperleft_x + bbx_width;
	bbx_points[2].y = upperleft_y + bbx_height;

	bbx_points[3].x = upperleft_x;
	bbx_points[3].y = upperleft_y + bbx_height;

	bbx_points[4] = bbx_points[0];

	bbx_center.x = (bbx_points[0].x + bbx_points[1].x) / 2;
	bbx_center.y = (bbx_points[0].y + bbx_points[2].y) / 2;

	return true;
}

bool FrontalFace15x20::bbx2ldm()
{
	warning("FrontalFace15x20::bbx2ldm() not implemented.");
	return false;
}

bool FrontalFace15x20::bbx2ldm(int n_bbx_points_, sPoint2D *bbx_points_, int n_ldm_points_, sPoint2D *ldm_points_)
{
	const bool verbose = getBOption("verbose");

	if(verbose)
		message("FrontalFace15x20::bbx2ldm(...)");

	if(n_ldm_points_ != 2)
	{
		warning("FrontalFace15x20::bbx2ldm() n_ldm_points_ != 2.");
		return false;
	}

	if(n_bbx_points_ != 4)
	{
		warning("FrontalFace15x20::to() n_bbx_points_ != 4.");
		return false;
	}

	double x = bbx_points_[0].x;
	double y = bbx_points_[0].y;
	double width_ = bbx_points_[1].x - bbx_points_[0].x;
	if((int)(width_) % 2 == 0) width_ += 1;
	double height_ = bbx_points_[2].y - bbx_points_[0].y;

	if(verbose)
	{
		print("> x = %g\n", x);
		print("> y = %g\n", y);
		print("> w = %g\n", width_);
		print("> h = %g\n", height_);
	}

	double Cx = x + width_/2.0;

	if(verbose)
		print("> Cx = %g\n", Cx);

	double z = PUPIL_SE * width_ / ZY_ZY;

	if(verbose)
		print("> z = %g\n", z);

	double Lx = Cx - z -1;
	double Rx = Cx + z -1;

	if(verbose)
	{
		print("> Lx = %g\n", Lx);
		print("> Rx = %g\n", Rx);
	}

	double Ry = y + height_ * (TR_GN - EN_GN) / TR_GN + 1;
	double Ly = Ry;

	ldm_points_[0].x = (int)(Lx);
	ldm_points_[1].x = (int)(Rx);

	ldm_points_[0].y = (int)(Ly);
	ldm_points_[1].y = (int)(Ry);

	if(verbose)
	{
		print("> l_eye = (%g, %g)\n", ldm_points_[0].x, ldm_points_[0].y);
		print("> r_eye = (%g, %g)\n", ldm_points_[1].x, ldm_points_[1].y);
	}

	return true;
}

FrontalFace15x20::~FrontalFace15x20()
{
}

}
