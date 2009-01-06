#include "FaceModel.h"
#include "File.h"

namespace Torch {

FaceModel::FaceModel(int postype_) : ShapeModel(5)
{
	// default loading format
	postype = postype_;

	//
	r_eye_idx = -1;
	l_eye_idx = -1;
	nosetip_idx = -1;
	chin_idx = -1;

	model_width = -1;
	model_height = -1;
	bbx_center.x = bbx_center.y = 0;
	bbx_width = -1;
	bbx_height = -1;
}

double FaceModel::getAngle()
{
   	if(postype == 0)
	{
		double angle = -atan2(ldm_points[1].y - ldm_points[0].y, ldm_points[1].x - ldm_points[0].x);

		return angle;
	}
	else if(r_eye_idx != -1 && l_eye_idx != -1)
	{
		double angle = -atan2(getReye().y - getLeye().y, getReye().x - getLeye().x);

		return angle;
	}
	else return 0.0;
}

sPoint2D FaceModel::getCenter()
{
	sPoint2D p;

	p.x = -1;
	p.y = -1;

	return p;
}


sPoint2D FaceModel::getReye()
{
   	if(r_eye_idx == -1) error("FaceModel::getReye() this landmark does not exist");

	return ldm_points[r_eye_idx];
}

sPoint2D FaceModel::getLeye()
{
   	if(l_eye_idx == -1) error("FaceModel::getLeye() this landmark does not exist");

	return ldm_points[l_eye_idx];
}

sPoint2D FaceModel::getNosetip()
{
   	if(nosetip_idx == -1) error("FaceModel::getNosetip() this landmark does not exist");

	return ldm_points[nosetip_idx];
}

sPoint2D FaceModel::getChin()
{
   	if(chin_idx == -1) error("FaceModel::getChin() this landmark does not exist");

	return ldm_points[chin_idx];
}

void FaceModel::loadFile(File *file)
{
	const bool verbose = getBOption("verbose");

   	n_ldm_points = 0;
	r_eye_idx = -1;
	l_eye_idx = -1;
	nosetip_idx = -1;
	chin_idx = -1;

	if(postype == 0)
	{
		delete[] ldm_points;
		ldm_points = new sPoint2D[4];

		if(verbose) message("reading bbx pos format.");

		// IDIAP format (eyes center)
		float x, y, w, h;

		file->scanf("%f", &x);
		file->scanf("%f", &y);
		file->scanf("%f", &w);
		file->scanf("%f", &h);

		ldm_points[n_ldm_points].x   = x;
		ldm_points[n_ldm_points++].y = y;
		ldm_points[n_ldm_points].x   = x + w;
		ldm_points[n_ldm_points++].y = y;
		ldm_points[n_ldm_points].x   = x + w;
		ldm_points[n_ldm_points++].y = y + h;
		ldm_points[n_ldm_points].x   = x;
		ldm_points[n_ldm_points++].y = y + w;

		if(verbose) message(" + BBX (%.1f-%.1f) - (%.1f-%.1f)", x, y, w, h);
	}
	else if(postype == 1)
	{
		delete[] ldm_points;
		ldm_points = new sPoint2D[2];

		if(verbose) message("reading eye centers pos format.");

		// IDIAP format (eyes center)
		float x, y;

		// left eye
		file->scanf("%f", &x);
		file->scanf("%f", &y);

		ldm_points[n_ldm_points].x = x;
		ldm_points[n_ldm_points++].y = y;

		// right eye
		file->scanf("%f", &x);
		file->scanf("%f", &y);

		ldm_points[n_ldm_points].x = x;
		ldm_points[n_ldm_points++].y = y;

		l_eye_idx = 0;
		r_eye_idx = 1;

		if(verbose) message(" + Eyes (%.1f-%.1f) - (%.1f-%.1f)", getLeye().x, getLeye().y, getReye().x, getReye().y);
	}
	else if(postype == 2)
	{
		delete[] ldm_points;
		ldm_points = new sPoint2D[10];

		if(verbose) message("reading Banca pos format.");

		// BANCA format
		float x, y;

		for(int i = 0 ; i < 10 ; i++)
		{
			file->scanf("%f", &x);
			file->scanf("%f", &y);
			ldm_points[n_ldm_points].x = x;
			ldm_points[n_ldm_points++].y = y;
		}

		l_eye_idx = 1;
		r_eye_idx = 4;

		if(verbose) message(" + Eyes (%.1f-%.1f) - (%.1f-%.1f)", getLeye().x, getLeye().y, getReye().x, getReye().y);
	}
	else if(postype == 3)
	{
		delete[] ldm_points;
		ldm_points = new sPoint2D[2];

		if(verbose) message("reading eye corners pos format.");

		// IDIAP format (eyes corner)
		float l_eye_l_x, l_eye_l_y;
		file->scanf("%f", &l_eye_l_x);
		file->scanf("%f", &l_eye_l_y);
		float l_eye_r_x, l_eye_r_y;
		file->scanf("%f", &l_eye_r_x);
		file->scanf("%f", &l_eye_r_y);

		ldm_points[n_ldm_points].x = (l_eye_l_x + l_eye_r_x) / 2.0;
		ldm_points[n_ldm_points++].y = (l_eye_l_y + l_eye_r_y) / 2.0;

		float r_eye_l_x, r_eye_l_y;
		file->scanf("%f", &r_eye_l_x);
		file->scanf("%f", &r_eye_l_y);
		float r_eye_r_x, r_eye_r_y;
		file->scanf("%f", &r_eye_r_x);
		file->scanf("%f", &r_eye_r_y);

		ldm_points[n_ldm_points].x = (r_eye_l_x + r_eye_r_x) / 2.0;
		ldm_points[n_ldm_points++].y = (r_eye_l_y + r_eye_r_y) / 2.0;

		l_eye_idx = 0;
		r_eye_idx = 1;

		if(verbose) message(" + Eyes (%.1f-%.1f) - (%.1f-%.1f)", getLeye().x, getLeye().y, getReye().x, getReye().y);
	}
	else if(postype == 4)
	{
		delete[] ldm_points;
		ldm_points = new sPoint2D[4];

		if(verbose) message("reading eye corners + nose tip + chin pos format (non frontal 22.5 degree).");

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

		ldm_points[n_ldm_points].x = (l_eye_l_x + l_eye_r_x) / 2.0;
		ldm_points[n_ldm_points++].y = (l_eye_l_y + l_eye_r_y) / 2.0;

		// right eye
		float r_eye_l_x, r_eye_l_y;
		file->scanf("%f", &r_eye_l_x);
		file->scanf("%f", &r_eye_l_y);
		float r_eye_r_x, r_eye_r_y;
		file->scanf("%f", &r_eye_r_x);
		file->scanf("%f", &r_eye_r_y);

		ldm_points[n_ldm_points].x = (r_eye_l_x + r_eye_r_x) / 2.0;
		ldm_points[n_ldm_points++].y = (r_eye_l_y + r_eye_r_y) / 2.0;

		// nose tip
		float x, y;
		file->scanf("%f", &x);
		file->scanf("%f", &y);

		ldm_points[n_ldm_points].x = x;
		ldm_points[n_ldm_points++].y = y;

		// chin
		file->scanf("%f", &x);
		file->scanf("%f", &y);

		ldm_points[n_ldm_points].x = x;
		ldm_points[n_ldm_points++].y = y;

		l_eye_idx = 0;
		r_eye_idx = 1;
		nosetip_idx = 2;
		chin_idx = 3;

		if(verbose)
		{
			message(" + Eyes (%.1f-%.1f) - (%.1f-%.1f)", getLeye().x, getLeye().y, getReye().x, getReye().y);
			message(" + Nosetip (%.1f-%.1f)", getNosetip().x, getNosetip().y);
			message(" + Chin (%.1f-%.1f)", getChin().x, getChin().y);
		}
	}
	else if(postype == 5)
	{
		delete[] ldm_points;
		ldm_points = new sPoint2D[4];

		if(verbose) message("reading left eye corners + right eye center + nose tip + chin pos format (non frontal 45/67.5 degree).");

		int n_;

		file->scanf("%d", &n_);

		if(n_ != 5) error("Number of points (%d) <> 5", n_);

		// left eye
		float l_eye_l_x, l_eye_l_y;
		file->scanf("%f", &l_eye_l_x);
		file->scanf("%f", &l_eye_l_y);
		float l_eye_r_x, l_eye_r_y;
		file->scanf("%f", &l_eye_r_x);
		file->scanf("%f", &l_eye_r_y);

		ldm_points[n_ldm_points].x = (l_eye_l_x + l_eye_r_x) / 2.0;
		ldm_points[n_ldm_points++].y = (l_eye_l_y + l_eye_r_y) / 2.0;

		// right eye
		float x, y;
		file->scanf("%f", &x);
		file->scanf("%f", &y);

		ldm_points[n_ldm_points].x = x;
		ldm_points[n_ldm_points++].y = y;

		// nose tip
		file->scanf("%f", &x);
		file->scanf("%f", &y);

		ldm_points[n_ldm_points].x = x;
		ldm_points[n_ldm_points++].y = y;

		// chin
		file->scanf("%f", &x);
		file->scanf("%f", &y);

		ldm_points[n_ldm_points].x = x;
		ldm_points[n_ldm_points++].y = y;

		l_eye_idx = 0;
		r_eye_idx = 1;
		nosetip_idx = 2;
		chin_idx = 3;

		if(verbose)
		{
			message(" + Eyes (%.1f-%.1f) - (%.1f-%.1f)", getLeye().x, getLeye().y, getReye().x, getReye().y);
			message(" + Nosetip (%.1f-%.1f)", getNosetip().x, getNosetip().y);
			message(" + Chin (%.1f-%.1f)", getChin().x, getChin().y);
		}
	}
	else if(postype == 6)
	{
		delete[] ldm_points;
		ldm_points = new sPoint2D[4];

		if(verbose) message("reading left eye center + nose tip + chin pos format  (non frontal 90 degree).");

		int n_;

		file->scanf("%d", &n_);

		if(n_ != 3) error("Number of points (%d) <> 3", n_);

		// left eye
		float x, y;
		file->scanf("%f", &x);
		file->scanf("%f", &y);

		ldm_points[n_ldm_points].x = x;
		ldm_points[n_ldm_points++].y = y;

		// nose tip
		file->scanf("%f", &x);
		file->scanf("%f", &y);

		ldm_points[n_ldm_points].x = x;
		ldm_points[n_ldm_points++].y = y;

		// chin
		file->scanf("%f", &x);
		file->scanf("%f", &y);

		ldm_points[n_ldm_points].x = x;
		ldm_points[n_ldm_points++].y = y;

		l_eye_idx = 0;
		nosetip_idx = 1;
		chin_idx = 2;

		if(verbose)
		{
			message(" + Left eye (%.1f-%.1f)", getLeye().x, getLeye().y);
			message(" + Nosetip (%.1f-%.1f)", getNosetip().x, getNosetip().y);
			message(" + Chin (%.1f-%.1f)", getChin().x, getChin().y);
		}
	}
	else if((postype >= 68) && (postype <= 71))
	{
		if(verbose) message("reading Tim Cootes Markup point format.");

		char buffer[250];
		do
		{
			file->gets(buffer, 250);
		} while(buffer[0] != '{');

		n_ldm_points = 68;
		delete[] ldm_points;
		ldm_points = new sPoint2D[68];

		for(int i = 0 ; i < 68 ; i++)
		{
		   	float x, y;
			file->scanf("%g", &x);
			file->scanf("%g", &y);
			ldm_points[i].x = x;
			ldm_points[i].y = y;

			if(verbose) print("> %g %g\n", x, y);

			l_eye_idx = 31;
			r_eye_idx = 36;

		}

		if(verbose) message(" + Eyes (%.1f-%.1f) - (%.1f-%.1f)", getLeye().x, getLeye().y, getReye().x, getReye().y);
	}
	else warning("FaceModel: Invalid postype (%d).", postype);
}

FaceModel::~FaceModel()
{
}

}

