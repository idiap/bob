#ifndef VECTOR3D_INC
#define VECTOR3D_INC

#include "Point3D.h"

namespace Torch {

	class Matrix3D;

	/** This class is designed to handle a vector in 3D

	    @author Sebastien Marcel (marcel@idiap.ch)
	    @author Dan Sunday (http://www.softsurfer.com/)
	    @version 2.1
	    \Date
	    @since 1.0
	*/
	class Vector3D : public Point3D
	{
	public:
		/** @name constructors 	*/
		//@{
		///
		Vector3D() : Point3D() {};

		///
		Vector3D(int a) : Point3D(a) {};

		///
		Vector3D(double a) : Point3D(a) {};

		///
		Vector3D(int a, int b) : Point3D(a,b) {};

		///
		Vector3D(double a, double b) : Point3D(a,b) {};

		///
		Vector3D(int a, int b, int c) : Point3D(a,b,c) {};

		///
		Vector3D(double a, double b, double c) : Point3D(a,b,c) {};
		//@}

		/// destructor
		~Vector3D() {};

		//-----

		/** @name Vector3D Unary Operations */
		//@{
		/// Unary minus
		Vector3D operator-();

		/// Unary 2D perp operator
		Vector3D operator~();
		//@}

		//-----

		/** @name Vector3D Arithmetic Operations */
		//@{
		/// vector add
		Vector3D operator+(const Vector3D&);

		/// vector subtract
		Vector3D operator-(const Vector3D&);

		/// inner dot product
		double operator*(const Vector3D&);

		/// 2D exterior perp product
		double operator|(const Vector3D&);

		/// 3D exterior cross product
		Vector3D operator^(const Vector3D&);

		/// vector scalar mult
		Vector3D& operator*=(double);

		/// vector scalar div
		Vector3D& operator/=(double);

		/// vector increment
		Vector3D& operator+=(const Vector3D&);

		/// vector decrement
		Vector3D& operator-=(const Vector3D&);

		/// 3D exterior cross product
		Vector3D& operator^=(const Vector3D&);
		//@}

		//-----

		/** @name Vector Properties */
		//@{
		/// vector length
		double len() { return sqrt(get(0)*get(0) + get(1)*get(1) + get(2)*get(2)); }

		/// vector length squared (faster)
		double len2() { return (get(0)*get(0) + get(1)*get(1) + get(2)*get(2)); }
		//@}

		//-----

		/// convert vector to unit length
		void normalize();
	};

	/** @name Point3D scalar multiplication and divisions

	    @author Sebastien Marcel (marcel@idiap.ch)
	    @author Dan Sunday (http://www.softsurfer.com/)
	    @version 2.1
	    \Date
	    @since 1.0
	*/
	//@{
	/// Vector3D operator*(int, Vector3D)
	Vector3D operator*(int, const Vector3D&);

	/// Vector3D operator*(double, Vector3D)
	Vector3D operator*(double, const Vector3D&);

	/// Vector3D operator*(Vector3D, int)
	Vector3D operator*(const Vector3D&, int);

	/// Vector3D operator*(Vector3D, double)
	Vector3D operator*(const Vector3D&, double);

	/// Vector3D operator/(Vector3D, int)
	Vector3D operator/(const Vector3D&, int);

	/// Vector3D operator/(Vector3D, double)
	Vector3D operator/(const Vector3D&, double);
	//@}

	/** @name vector 3D sums

	    @author Dan Sunday (http://www.softsurfer.com/)
	    @version 2.0
	    \Date
	    @since 1.0
	*/
	//@{
	/// vector sum from int
	Vector3D sum(int, int *, const Vector3D *);

	/// vector sum from double
	Vector3D sum(int, double *, const Vector3D *);
	//@}

}

#endif
