#include "ip/RotationMatrix2D.h"

namespace Torch {

double ccwAngle(double degAngle_)
{
   	double degAngle = degAngle_;

   	if(degAngle >= 360.0)
	{
	   	degAngle -= 360.0;
		while(degAngle >= 360.0) degAngle -= 360.0;
	}
	else if(degAngle < 0.0)
	{
	   	degAngle += 360.0;
    		while(degAngle < 0.0) degAngle += 360.0;
	}

	return degAngle;
}

void RotationMatrix2D::computeSinCosin()
{
	if(degAngle == 0.0)
	{
		radAngle = 0.0;
		dSin = 0.0;
		dCos = 1.0;
		dSinOpposite = 0.0;
		dCosOpposite = 1.0;
	}
	else if(degAngle == 180.0)
	{
		radAngle = M_PI;
		dSin = 0.0;
		dCos = -1.0;
		dSinOpposite = 0.0;
		dCosOpposite = -1.0;
	}
	else if(degAngle == 90.0)
	{
		radAngle = M_PI_2;
		dSin = 1.0;
		dCos = 0.0;
		dSinOpposite = -1.0;
		dCosOpposite = 0.0;
	}
	else if(degAngle == 270.0)
	{
		radAngle = 3.0 * M_PI_2;
		dSin = -1.0;
		dCos = 0.0;
		dSinOpposite = 1.0;
		dCosOpposite = 0.0;
	}
	else
	{
		// convert degrees to radians
		radAngle = degAngle * M_PI / 180.0;
		dSin = sin(radAngle);
		dCos = cos(radAngle);
		dSinOpposite = sin(-radAngle);
		dCosOpposite = cos(-radAngle);
	}

	//print("Sin = %g\t Cos = %g\n", dSin, dCos);

	dTan2 = tan(radAngle/2.0);

	xc = yc = 0.0;
	XC = YC = 0.0;
}

RotationMatrix2D::RotationMatrix2D(double degAngle_)
{
   	//
	degAngle = ccwAngle(degAngle_);

	//print("angle = %g\n", degAngle);

	//
	computeSinCosin();

	set(0, 0, dCos); set(0, 1, -dSin);
	set(1, 0, dSin); set(1, 1, dCos);
}

RotationMatrix2D::RotationMatrix2D(double degAngle_, double scale_)
{
   	//
	degAngle = ccwAngle(degAngle_);

	//
	computeSinCosin();

	set(0, 0, dCos * scale_); set(0, 1, -dSin * scale_);
	set(1, 0, dSin * scale_); set(1, 1, dCos * scale_);
}

/// Reset to the given angle
void RotationMatrix2D::reset(double degAngle_)
{
	//
	degAngle = ccwAngle(degAngle_);

	//
	computeSinCosin();

	set(0, 0, dCos); set(0, 1, -dSin);
	set(1, 0, dSin); set(1, 1, dCos);
}

RotationMatrix2D::~RotationMatrix2D()
{
}

void rotate(int x, int y, int *X, int *Y, const RotationMatrix2D *rim)
{
	double xo, yo;
	double xr, yr;

   	// Changing coordinate system from src image to unity circle
   	xo = (double) x - rim->xc;
	yo = (double) y - rim->yc;

	// Performing rotation
	xr = rim->dCos * xo - rim->dSin * yo;
	yr = rim->dSin * xo + rim->dCos * yo;

   	// Changing coordinate system from unity circle to dest image
  	*X = (int) (xr + rim->XC);
	*Y = (int) (yr + rim->YC);
}

void rotate(double x, double y, double *X, double *Y, const RotationMatrix2D *rim)
{
	double xo, yo;
	double xr, yr;

   	// Changing coordinate system from src image to unity circle
   	xo = x - rim->xc;
	yo = y - rim->yc;

	// Performing rotation
	xr = rim->dCos * xo - rim->dSin * yo;
	yr = rim->dSin * xo + rim->dCos * yo;

   	// Changing coordinate system from unity circle to dest image
  	*X = xr + rim->XC;
	*Y = yr + rim->YC;
}

void unrotate(int x, int y, int *X, int *Y, const RotationMatrix2D *rim)
{
	double xo, yo;
	double xr, yr;

   	// Changing coordinate system from dst image to unity circle
   	xo = (double) x - rim->XC;
	yo = (double) y - rim->YC;

	// Performing rotation
	xr = rim->dCosOpposite * xo - rim->dSinOpposite * yo;
	yr = rim->dSinOpposite * xo + rim->dCosOpposite * yo;

   	// Changing coordinate system from unity circle to src image
  	*X = FixI (xr + rim->xc);
	*Y = FixI (yr + rim->yc);
}

void unrotate(double x, double y, double *X, double *Y, const RotationMatrix2D *rim)
{
	double xo, yo;
	double xr, yr;

   	// Changing coordinate system from dst image to unity circle
   	xo = x - rim->XC;
	yo = y - rim->YC;

	// Performing rotation
	xr = rim->dCosOpposite * xo - rim->dSinOpposite * yo;
	yr = rim->dSinOpposite * xo + rim->dCosOpposite * yo;

   	// Changing coordinate system from unity circle to src image
  	*X = xr + rim->xc;
	*Y = yr + rim->yc;
}


}
