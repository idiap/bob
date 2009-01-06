#ifndef ROTATION_MATRIX2D_INC
#define ROTATION_MATRIX2D_INC

#include "Matrix2D.h"

namespace Torch {

/** This class is designed to handle a rotation matrix in 2D

    @author Sebastien Marcel (marcel@idiap.ch)
    @version 2.0
    \Date
    @since 1.0
*/
class RotationMatrix2D : public Matrix2D
{
	// computes internal sine and cosine
	void computeSinCosin();
	
public:
   	/// angle in degrees
	double degAngle;
	
	/// angle in radians
	double radAngle;
	
	/// tan of radAngle/2
	double dTan2;

	/// sin of radAngle
	double dSin;
	
	/// cos of radAngle
	double dCos;
	
	/// sin of -radAngle
	double dSinOpposite;
	
	/// cos of -radAngle
	double dCosOpposite;
	
	/** @name coordinate system*/
	//@{
	///
	double xc;

	///
	double yc;
	
	///
	double XC;
	
	///
	double YC;
	//@}

	//-----
	
	/** @name constructor */
	//@{
	/// Computes the rotation matrix for a given angle
	RotationMatrix2D(double degAngle_);

	/// Computes the rotation/scale matrix for a given angle and scale
	RotationMatrix2D(double degAngle_, double scale_);
	//@}
	
	/// destructor
	~RotationMatrix2D();
};

/** This function bounds an angle in degrees

    @param degAngle_ is any angle in degree
    @return an angle in degree between $0$ and $360$
*/
double ccwAngle(double degAngle_);

/** @name rotate/unrotate a 2D point */
//@{
/** rotate a 2D point (integer)

    @param x_ is the #x# coordinate to rotate
    @param y_ is the #y# coordinate to rotate
    @param X_ is the #x# coordinate rotated
    @param Y_ is the #y# coordinate rotated
    @param rim_ is the rotation matrix
*/
void rotate(int x_, int y_, int *X_, int *Y_, RotationMatrix2D *rim_);

/** rotate a 2D point (double)

    @param x_ is the #x# coordinate to rotate
    @param y_ is the #y# coordinate to rotate
    @param X_ is the #x# coordinate rotated
    @param Y_ is the #y# coordinate rotated
    @param rim_ is the rotation matrix
*/
void rotate(double x_, double y_, double *X_, double *Y_, RotationMatrix2D *rim_);

/** rotate a 2D point (integer) in the opposite direction

    @param x_ is the #x# coordinate to rotate
    @param y_ is the #y# coordinate to rotate
    @param X_ is the #x# coordinate rotated
    @param Y_ is the #y# coordinate rotated
    @param rim_ is the rotation matrix
*/
void unrotate(int x_, int y_, int *X_, int *Y_, RotationMatrix2D *rim_);

/** rotate a 2D point (double) in the opposite direction

    @param x_ is the #x# coordinate to rotate
    @param y_ is the #y# coordinate to rotate
    @param X_ is the #x# coordinate rotated
    @param Y_ is the #y# coordinate rotated
    @param rim_ is the rotation matrix
*/
void unrotate(double x_, double y_, double *X_, double *Y_, RotationMatrix2D *rim_);
//@}

}

#endif
