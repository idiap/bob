#include "NonFrontalFace19x19.h"

namespace Torch {

NonFrontalFace19x19::NonFrontalFace19x19(int pan_, int postype_) : FaceModel(postype_)
{
   	pan = pan_;

	D_EYES = 10.0;
	Y_UPPER = 5.0;
	EYE_CHIN = 16.0;

	model_width = 19.0;
	model_height = 19.0;
}

bool NonFrontalFace19x19::ldm2bbx()
{
	const bool verbose = getBOption("verbose");

	int upperleft_x = 0;
	int upperleft_y = 0;
	sPoint2D l_eye;
	sPoint2D r_eye;
	sPoint2D nosetip;
	sPoint2D chin;
	double c0x, c0y, EEx, EEy;
	double EC, ratio;

	switch(pan)
	{
	case 0: // Frontal face

		l_eye = getLeye();
		r_eye = getReye();

		// Distance between the eyes
		EEx = r_eye.x - l_eye.x;
		EEy = r_eye.y - l_eye.y;

		// The face is non oriented
		if(!IS_NEAR(EEy, 0, 0.001))
	   		warning("NonFrontalFace19x19: y coordinates of eyes should be equal");

		c0x = l_eye.x + EEx / 2.0;
		c0y = l_eye.y + EEy / 2.0;

		if(verbose)
			print("NonFrontalFace19x19: c0x=%g c0y=%g\n", c0x, c0y);

		ratio = EEx / D_EYES;

		bbx_width = (int) (ratio * model_width + 0.5);
		bbx_height = bbx_width;

		if(bbx_width < 19)
			warning("NonFrontalFace19x19::ldm2bbx() width=%d < 19", bbx_width);

		/* The upperleft x coordinate of the face is given
	   		by the middle of the face minus the half size of the face */
		upperleft_x = (int) (c0x - (ratio * model_width / 2.0));

		/* The upperleft y coordinate of the face is given
	   		by the constants #en_gn# and #tr_gn# */
		upperleft_y = (int) (c0y - (ratio * Y_UPPER));

		break;

	case 1: // Non-frontal face (22.5 degree)

		l_eye = getLeye();
		r_eye = getReye();
		nosetip = getNosetip();
		chin = getChin();

		// Distance between the eyes
		EEx = r_eye.x - l_eye.x;
		EEy = r_eye.y - l_eye.y;

		// The center of the eyes is taken as the center
		// in x of the bounding box
		c0x = l_eye.x + EEx / 2.0;
		c0y = l_eye.y + EEy / 2.0;

		EC = chin.y - l_eye.y;
		ratio = EC / EYE_CHIN;

		bbx_width = (int) (ratio * model_width + 0.5);
		bbx_height = bbx_width;

		if(bbx_width < 19)
			warning("NonFrontalFace19x19::ldm2bbx() width=%d < 19", bbx_width);

		upperleft_x = (int) (c0x - (ratio * model_width / 2.0));
		upperleft_y = (int) (c0y - (ratio * Y_UPPER));

		if(upperleft_x < 0)
			warning("NonFrontalFace19x19::ldm2bbx() upperleft_x=%d < 0 !", upperleft_x);

		break;

	case 2: // Non-frontal face (45 degree)

		l_eye = getLeye();
		r_eye = getReye();
		nosetip = getNosetip();
		chin = getChin();

		// Distance between the eyes
		EEx = r_eye.x - l_eye.x;
		EEy = r_eye.y - l_eye.y;

		// The center of the eyes is shifted to the left and
		// then taken as the center in x of the bounding box
		c0x = l_eye.x + EEx / 3.0;
		c0y = l_eye.y + EEy / 2.0;

		EC = chin.y - l_eye.y;
		ratio = EC / EYE_CHIN;

		bbx_width = (int) (ratio * model_width + 0.5);
		bbx_height = bbx_width;

		if(bbx_width < 19)
			warning("NonFrontalFace19x19::ldm2bbx() width=%d < 19", bbx_width);

		upperleft_x = (int) (c0x - (ratio * model_width / 2.0));
		upperleft_y = (int) (c0y - (ratio * Y_UPPER));

		if(upperleft_x < 0)
			warning("NonFrontalFace19x19::ldm2bbx() upperleft_x=%d < 0 !", upperleft_x);

		break;

	case 3: // Non-frontal face (67.5 degree)

		l_eye = getLeye();
		r_eye = getReye();
		nosetip = getNosetip();
		chin = getChin();

		// Distance between the eyes
		EEx = r_eye.x - l_eye.x;
		EEy = r_eye.y - l_eye.y;

		// The center of the eyes is shifted to the left and
		// then taken as the center in x of the bounding box
		c0x = l_eye.x + EEx / 4.0;
		c0y = l_eye.y + EEy / 2.0;

		EC = chin.y - l_eye.y;
		ratio = EC / EYE_CHIN;

		bbx_width = (int) (ratio * model_width + 0.5);
		bbx_height = bbx_width;

		if(bbx_width < 19)
			warning("NonFrontalFace19x19::ldm2bbx() width=%d < 19", bbx_width);

		upperleft_x = (int) (c0x - (ratio * model_width / 2.0));
		upperleft_y = (int) (c0y - (ratio * Y_UPPER));

		if(upperleft_x < 0)
			warning("NonFrontalFace19x19::ldm2bbx() upperleft_x=%d < 0 !", upperleft_x);

		break;

	case 4: // Non-frontal face (90 degree)

		l_eye = getLeye();
		nosetip = getNosetip();
		chin = getChin();

		EC = chin.y - l_eye.y;
		ratio = EC / EYE_CHIN;

		bbx_width = (int) (ratio * model_width + 0.5);
		bbx_height = bbx_width;

		if(bbx_width < 19)
			warning("NonFrontalFace19x19::ldm2bbx() width=%d < 19", bbx_width);

		upperleft_x = (int) (nosetip.x + (ratio * 1.0) - bbx_width);
		upperleft_y = (int) (l_eye.y - (ratio * Y_UPPER));

		if(upperleft_x < 0)
			warning("NonFrontalFace19x19::ldm2bbx() upperleft_x=%d < 0 !", upperleft_x);

		break;

	default:
		warning("NonFrontalFace19x19::ldm2bbx() pan (%d) not supported", pan);

	}

	if(verbose)
		print("NonFrontalFace19x19: x=%d y=%d w=%d h=%d\n", upperleft_x, upperleft_y, bbx_width, bbx_height);

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

bool NonFrontalFace19x19::bbx2ldm()
{
	warning("NonFrontalFace19x19::bbx2ldm() not implemented.");
	return false;
}

bool NonFrontalFace19x19::bbx2ldm(int n_bbx_points_, sPoint2D *bbx_points_, int n_ldm_points_, sPoint2D *ldm_points_)
{
	if(n_bbx_points_ != 4)
	{
		warning("NonFrontalFace19x19::bbx2ldm() n_bbx_points_ != 4.");
		return false;
	}

	double Rx, Ry, Lx, Ly, Nx, Ny;

	double x = bbx_points_[0].x;
	double y = bbx_points_[0].y;
	double width_ = bbx_points_[1].x - bbx_points_[0].x;
	double ratio = width_ / model_width;
	switch(abs(pan))
	{
	case 0: // Frontal face
	case 1: // Non-frontal face (+/-22.5 degree)
		if (pan > 0) //(+22.5 degree)
		{
			Rx = ratio * (D_EYES + model_width) / 2.0 + x;
			Lx = Rx - D_EYES * ratio;

			Ry = y + ratio * Y_UPPER;
			Ly = Ry;
		}

		else //(-22.5 degree)
		{
			Lx = ratio * (D_EYES + model_width) / 2.0 + x;
			Rx = Lx - D_EYES * ratio;

			Ry = y + ratio * Y_UPPER;
			Ly = Ry;
		}

		ldm_points_[0].x = FixI(Lx);
		ldm_points_[1].x = FixI(Rx);

		ldm_points_[0].y = FixI(Ly);
		ldm_points_[1].y = FixI(Ry);

		break;

	case 2: // Non-frontal face (+/-45 degree)

		if (pan > 0) //(+45 degree)
		{
			Rx = ratio * (D_EYES + model_width) / 1.9 + x;
			Lx = Rx - ratio * D_EYES / 1.1;

			Ry = y + ratio * Y_UPPER;
			Ly = Ry;
		}

		else //(-45 degree)
		{
			Lx = ratio * (D_EYES + model_width) / 2.0 + x;
			Rx = Lx - ratio * D_EYES / 1.1;

			Ry = y + ratio * Y_UPPER;
			Ly = Ry;
		}

		ldm_points_[0].x = FixI(Lx);
		ldm_points_[1].x = FixI(Rx);

		ldm_points_[0].y = FixI(Ly);
		ldm_points_[1].y = FixI(Ry);

		break;

	case 3: // Non-frontal face (+/-67.5 degree)

		if (pan > 0) //(+67.5 degree)
		{
			Rx = ratio * (D_EYES + model_width) / 1.8 + x;
			Lx = Rx - ratio * D_EYES / 1.2;

			Ry = y + ratio * Y_UPPER;
			Ly = Ry;
		}

		else //(-67.5 degree)
		{
			Lx = ratio * (D_EYES + model_width) / 2.3 + x;
			Rx = Lx - ratio * D_EYES / 1.1;

			Ry = y + ratio * Y_UPPER;
			Ly = Ry;
		}

		ldm_points_[0].x = FixI(Lx);
		ldm_points_[1].x = FixI(Rx);

		ldm_points_[0].y = FixI(Ly);
		ldm_points_[1].y = FixI(Ry);

		break;

	case 4: // Non-frontal face (+/-90 degree)

		if (pan > 0) //(+90 degree)
		{
			Nx = x + width_ - ratio;
			Ly = y + ratio * Y_UPPER;

			Lx = Nx - ratio * 5.5;
			Ny = Ly + ratio * 4.75 ;
		}

		else //(-90 degree)
		{
			Nx = x + ratio;
			Ly = y + ratio * Y_UPPER;

			Lx = Nx + ratio * 5.5;
			Ny = Ly + ratio * 4.75 ;
		}

		ldm_points_[0].x = FixI(Lx);
		ldm_points_[1].x = FixI(Nx);

		ldm_points_[0].y = FixI(Ly);
		ldm_points_[1].y = FixI(Ny);

		break;
	}

	return true;
}

double NonFrontalFace19x19::getAngle()
{
	double angle = 0.0;

	message("NonFrontalFace19x19::getAngle()");

	switch(pan)
	{
	case 0: // Frontal face

   		if(r_eye_idx != -1 && l_eye_idx != -1)
			angle = -atan2(getReye().y - getLeye().y, getReye().x - getLeye().x);

		break;

	case 1: // Non-frontal face (22.5 degree)

	   	if(r_eye_idx != -1 && l_eye_idx != -1)
			angle = -atan2(getReye().y - getLeye().y, (getReye().x - getLeye().x) / cos(3.1415/8.0));

		break;

	case 2: // Non-frontal face (45 degree)

   		if(r_eye_idx != -1 && l_eye_idx != -1)
			angle = -atan2(getReye().y - getLeye().y, (getReye().x - getLeye().x) / cos(3.1415/4.0));

		break;

	case 3: // Non-frontal face (67.5 degree)

		if(r_eye_idx != -1 && l_eye_idx != -1)
			angle = -atan2(getReye().y - getLeye().y, (getReye().x - getLeye().x) / cos(3.1415/4.0));

		break;

	case 4: // Non-frontal face (90 degree)

		break;
	}

	return angle;
 }

sPoint2D NonFrontalFace19x19::getCenter()
{
	sPoint2D p;

	p.x = -1;
	p.y = -1;

	message("NonFrontalFace19x19::getCenter()");

	switch(pan)
	{
	case 0: // Frontal face

	case 1: // Non-frontal face (22.5 degree)

	case 2: // Non-frontal face (45 degree)

	case 3: // Non-frontal face (67.5 degree)

		p.x = (getReye().x + getLeye().x) / 2.0;
		p.y = (getReye().y + getLeye().y) / 2.0;

		break;

	case 4: // Non-frontal face (90 degree)

		break;
	}

	return p;
}



NonFrontalFace19x19::~NonFrontalFace19x19()
{
}

}
