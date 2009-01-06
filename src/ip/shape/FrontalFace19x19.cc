#include "FrontalFace19x19.h"

namespace Torch {

FrontalFace19x19::FrontalFace19x19(int postype_) : FaceModel(postype_)
{
	D_EYES = 10.0;
	Y_UPPER = 5.0;

	model_width = 19.0;
	model_height = 19.0;
}

bool FrontalFace19x19::ldm2bbx()
{
	const bool verbose = getBOption("verbose");

	double c0x, c0y, EEx, EEy;

	sPoint2D l_eye = getLeye();
	sPoint2D r_eye = getReye();

	EEx = r_eye.x - l_eye.x;
	EEy = r_eye.y - l_eye.y;

	// The face is non oriented
	if(!IS_NEAR(EEy, 0, 0.001))
	   	warning("FrontalFace19x19: y coordinates of eyes should be equal");

	c0x = l_eye.x + EEx / 2.0;
	c0y = l_eye.y + EEy / 2.0;

	if(verbose)
		print("FrontalFace19x19: c0x=%g c0y=%g\n", c0x, c0y);

	double ratio = EEx / D_EYES;

	bbx_width = (int) (ratio * model_width + 0.5);
	bbx_height = bbx_width;

	if(bbx_width < 19)
		warning("FrontalFace19x19::fromEyesCenter() width=%d < 19", bbx_width);

	int upperleft_x;
	int upperleft_y;

	/* The upperleft x coordinate of the face is given
	   by the middle of the face minus the half size of the face */
	upperleft_x = (int) (c0x - (ratio * model_width / 2.0));

	/* The upperleft y coordinate of the face is given
	   by the constants #en_gn# and #tr_gn# */
	upperleft_y = (int) (c0y - (ratio * Y_UPPER));

	if(verbose)
		print("FrontalFace19x19: x=%d y=%d w=%d h=%d\n", upperleft_x, upperleft_y, bbx_width, bbx_height);

	// Filling bbx
	bbx_points[0].x = upperleft_x;
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

bool FrontalFace19x19::bbx2ldm()
{
	double x = bbx_points[0].x;
	double y = bbx_points[0].y;
	double width_ = bbx_points[1].x - bbx_points[0].x;
	double ratio = width_ / model_width;

	double Rx = ratio * (D_EYES + model_width) / 2 + x;
	double Lx = Rx - D_EYES * ratio;

	double Ry = y + ratio * Y_UPPER;
	double Ly = Ry;

	ldm_points[0].x = FixI(Lx);
	ldm_points[1].x = FixI(Rx);

	ldm_points[0].y = FixI(Ly);
	ldm_points[1].y = FixI(Ry);

	return true;
}

bool FrontalFace19x19::bbx2ldm(int n_bbx_points_, sPoint2D *bbx_points_, int n_ldm_points_, sPoint2D *ldm_points_)
{
	if(n_ldm_points_ != 2)
	{
		warning("FrontalFace19x19::bbx2ldm() n_ldm_points_ != 2.");
		return false;
	}

	if(n_bbx_points_ != 4)
	{
		warning("FrontalFace19x19::to() n_bbx_points_ != 4.");
		return false;
	}

	double x = bbx_points_[0].x;
	double y = bbx_points_[0].y;
	double width_ = bbx_points_[1].x - bbx_points_[0].x;
	double ratio = width_ / model_width;

	double Rx = ratio * (D_EYES + model_width) / 2 + x;
	double Lx = Rx - D_EYES * ratio;

	double Ry = y + ratio * Y_UPPER;
	double Ly = Ry;

	ldm_points_[0].x = FixI(Lx);
	ldm_points_[1].x = FixI(Rx);

	ldm_points_[0].y = FixI(Ly);
	ldm_points_[1].y = FixI(Ry);

	return true;
}

FrontalFace19x19::~FrontalFace19x19()
{
}

}
