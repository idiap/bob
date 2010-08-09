#ifndef POINT2D_INC
#define POINT2D_INC

#include "geomObject.h"
#include "trigonometry.h"
#include "core/Tensor.h"

namespace Torch {

	//-----------------------------------
	// Error codes
	#define Enot 0 // no error
	#define Edim 1 // dim of point invalid for operation
	#define Esum 2 // sum not affine (cooefs add to 1)

	// utility macros
	#define	abs(x)		((x) >= 0 ? x : -(x))
	#ifndef min
	#define	min(x,y)	((x) < (y) ? (x) : (y))
	#endif
	#ifndef max
	#define	max(x,y)	((x) > (y) ? (x) : (y))
	#endif

	class Vector2D;
	class Matrix2D;

	/** This class is designed to handle a point in 2D
		stored as a 1D DoubleTensor

	    @author Sebastien Marcel (marcel@idiap.ch)
	    @author Dan Sunday (http://www.softsurfer.com/)
	    @version 2.0
	    \date
	    @since 1.0
	*/
	class Point2D : public DoubleTensor
	{
	public:
		//-----

		/** @name Lots of Constructors */
		//@{
		///
		Point2D();

		///
		Point2D(int a);

		///
		Point2D(double a);

		///
		Point2D(int a, int b);

		///
		Point2D(double a, double b);

		///
		Point2D(const sPoint2D& p);

		///
		Point2D(const sPoint2D *p);
		//@}

		/// destructor
		virtual ~Point2D() {};

		//-----

		/** @name reset functions */
		//@{
		///
		void reset(int a, int b) { set(0, a); set(1, b); }

		///
		void reset(double a, double b) { set(0, a); set(1, b); }

		///
		void reset(const sPoint2D& p) { set(0, p.x); set(1, p.y); }
		//@}

		//-----

		/// draw the point onto an image
		virtual void draw(Image *image, Color color);

		//-----

		/// equality
		int operator==(const Point2D&);

		/// difference
		int operator!=(const Point2D&);

		//-----

		/** @name Point2D and Vector2D operations

		    These operations are not always valid. */
		//@{
		/// Vector difference
		Vector2D operator-(const Point2D&);

		/// Vector translate +
		Point2D  operator+(const Vector2D&);

		/// Vector translate -
		Point2D  operator-(const Vector2D&);

		/// Vector inc translate
		Point2D& operator+=(const Vector2D&);

		/// Vector dec translate
		Point2D& operator-=(const Vector2D&);
		//@}

		//-----

		/** @name collinearity conditions */
		//@{
		/// is On line (char= flag)
		bool isOnLine(const Point2D&, const Point2D&, char);

		/// is On line (flag= all)
		bool isOnLine(const Point2D&, const Point2D&);

		/// is On line (flag= before)
		bool isBefore(const Point2D&, const Point2D&);

		/// is On line (flag= between)
		bool isBetween(const Point2D&, const Point2D&);

		/// is On line (flag= after)
		bool isAfter(const Point2D&, const Point2D&);

		/// is On line (flag= between|after)
		bool isOnRay(const Point2D&, const Point2D&);
		//@}

		//-----

		/// tests if the point is Left|On|Right of an infinite line
		double isLeft(const Point2D&, const Point2D&);

		/// computes the surface of the triangle
		double Area(const Point2D&, const Point2D&);

		//-----

		/// return the 2D point
		sPoint2D getPoint();

		/// fix xy coordinates to integer values
		void fixI();
	};

	//-----

	/** @name Point2D scalar multiplication and divisions

	    \begin{verbatim}
	    Point2D Scalar Operations (convenient but often illegal)
	    using any type of scalar (int, float, or double)
	    are not valid for points in general,
	    unless they are 'affine' as coeffs of
	    a sum in which all the coeffs add to 1,
	    such as: the sum (a*P + b*Q) with (a+b == 1).

	    The programmer must enforce this (if they want to).
	    \end{verbatim}

	    @author Dan Sunday (http://www.softsurfer.com/)
	    @version 2.1
	    \date
	    @since 1.0
	*/
	//@{
	/// Point2D operator*(int, Point2D)
	Point2D operator*(int, const Point2D&);

	/// Point2D operator*(double, Point2D)
	Point2D operator*(double, const Point2D&);

	/// Point2D operator*(Point2D, int)
	Point2D operator*(const Point2D&, int);

	/// Point2D operator*(Point2D, double)
	Point2D operator*(const Point2D&, double);

	/// Point2D operator/(Point2D, int)
	Point2D operator/(const Point2D&, int);

	/// Point2D operator/(Point2D, double)
	Point2D operator/(const Point2D&, double);
	//@}

	//-----

	/** @name Point2D addition

	    Point2D addition (also convenient but often illegal)
	    is not valid unless part of an affine sum.

	    The programmer must enforce this (if they want to).

	    @author Dan Sunday (http://www.softsurfer.com/)
	    @version 2.1
	    \date
	    @since 1.0
	*/
	//@{
	/// Point2D operator+(Point2D, Point2D)
	Point2D operator+(const Point2D&, const Point2D&);
	//@}

	//-----

	/** @name Point2D rotation/scale

	    @author Sebastien Marcel (marcel@idiap.ch)
	    @version 2.1
	    \date
	    @since 1.0
	*/
	//@{
	/// Point2D operator*(Point2D, Matrix2D)
	Point2D operator*(const Point2D&, const Matrix2D&);
	//@}

	//-----

	/** @name Point2D affine sums

	    Check if coefficents sum to 1 and compute the weighted sum (even when not affine).

	    @author Dan Sunday (http://www.softsurfer.com/)
	    @version 2.0
	    \date
	    @since 1.0
	*/
	//@{
	/// affine sum from int
	Point2D asum(int, int *, const Point2D *);

	/// affine sum from double
	Point2D asum(int, double *, const Point2D *);
	//@}

	//-----

	/** @name Point2D distances

	    @author Dan Sunday (http://www.softsurfer.com/)
	    @version 2.0
	    \date
	    @since 1.0
	*/
	//@{
	/** Euclidean distance

	    computes the Euclidean distance $d$ between $P(x_1,y_1)$ and $Q(x_2, y_2)$ $d = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}$
	*/
	double d(const Point2D& P, const Point2D& Q);

	/** Squared Euclidean distance.

	    computes the Squared Euclidean distance $d$ between $P(x_1,y_1)$ and $Q(x_2, y_2)$ $d = (x_1 - x_2)^2 + (y_1 - y_2)^2$
	*/
	double d2(const Point2D& P, const Point2D& Q);
	//@}

	//-----

	/** @name Point2D angles

	    @author Dan Sunday (http://www.softsurfer.com/)
	    @version 2.0
	    \date
	    @since 1.0
	*/
	//@{
	/// return the angle (in radian) of a line (P1,P2) with the horizontal in P1
	double angle(const Point2D& P1, const Point2D& P2);

	/// return the angle (in radian) of 2 lines (P0,P1) and (P0,P2)
	double angle(const Point2D& P0, const Point2D& P1, const Point2D& B);
	//@}

	//-----

	/** @name Point2D relations */
	//@{
	/** test if a point is Left|On|Right of an infinite line.

	    You can find the algorithm of this function \URL[here]{http://geometryalgorithms.com/Archive/algorithm_0103/algorithm_0103.htm}.

	    @return >0 for P2 left of the line through P0 and P1
	    @return =0 for P2 on the line
	    @return <0 for P2 right of the line

	    @author Dan Sunday (http://www.softsurfer.com/)
	    @version 2.0
	    \date
	    @since 1.0
	*/
	int isLeft(const Point2D& P0, const Point2D& P1, const Point2D& P2);

	/** test the orientation 2D of a triangle.

	    It returns the angle direction of vectors (V0,V1) and (V0,V2).

	    @param V0 is the first vertex point
	    @param V1 is the second vertex point
	    @param V2 is the third vertex point
	    @return >0 for counterclockwise (ccw)
	    @return =0 for none (degenerate)
	    @return <0 for clockwise

	    @author Sebastien Marcel (marcel@idiap.ch)
	    @version 2.0
	    \date
	    @since 1.0
	*/
	int ccw(const Point2D& V0, const Point2D& V1, const Point2D& V2);
	//@}

}

#endif
