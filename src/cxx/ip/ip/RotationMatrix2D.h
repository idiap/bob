/**
 * @file cxx/ip/ip/RotationMatrix2D.h
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef ROTATION_MATRIX2D_INC
#define ROTATION_MATRIX2D_INC

#include "ip/Matrix2D.h"

namespace bob {

	/** This class is designed to handle a rotation matrix in 2D

	    @author Sebastien Marcel (marcel@idiap.ch)
	    @version 2.0
	    \date
	    @since 1.0
	*/
	class RotationMatrix2D : public Matrix2D
	{
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
		RotationMatrix2D(double degAngle_ = 0.0);

		/// Computes the rotation/scale matrix for a given angle and scale
		RotationMatrix2D(double degAngle_, double scale_ = 1.0);
		//@}

		/// Reset to the given angle
		void reset(double degAngle_);

		/// destructor
		~RotationMatrix2D();

	private:

		// computes internal sine and cosine
		void computeSinCosin();
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
	void rotate(int x_, int y_, int *X_, int *Y_, const RotationMatrix2D *rim_);

	/** rotate a 2D point (double)

	    @param x_ is the #x# coordinate to rotate
	    @param y_ is the #y# coordinate to rotate
	    @param X_ is the #x# coordinate rotated
	    @param Y_ is the #y# coordinate rotated
	    @param rim_ is the rotation matrix
	*/
	void rotate(double x_, double y_, double *X_, double *Y_, const RotationMatrix2D *rim_);

	/** rotate a 2D point (integer) in the opposite direction

	    @param x_ is the #x# coordinate to rotate
	    @param y_ is the #y# coordinate to rotate
	    @param X_ is the #x# coordinate rotated
	    @param Y_ is the #y# coordinate rotated
	    @param rim_ is the rotation matrix
	*/
	void unrotate(int x_, int y_, int *X_, int *Y_, const RotationMatrix2D *rim_);

	/** rotate a 2D point (double) in the opposite direction

	    @param x_ is the #x# coordinate to rotate
	    @param y_ is the #y# coordinate to rotate
	    @param X_ is the #x# coordinate rotated
	    @param Y_ is the #y# coordinate rotated
	    @param rim_ is the rotation matrix
	*/
	void unrotate(double x_, double y_, double *X_, double *Y_, const RotationMatrix2D *rim_);
	//@}

}

#endif
