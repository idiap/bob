#ifndef COMPLEX_INC
#define COMPLEX_INC

#include "vision.h"

namespace Torch {

/** This class is designed to handle complex numbers

    @author Sebastien Marcel (marcel@idiap.ch)
    @version 2.0
    \Date
    @since 2.0
*/
class Complex
{
public:
	/// real part
	double r;

	/// imaginary part
	double i;

	//-----

	/** @name Lots of Constructors */
	//@{
	///
	Complex() { r=i=0.0; }

	///
	Complex(double r_, double i_) { r=r_; i=i_; }

	//@}

	/// destructor
	virtual ~Complex() {};

	//-----

	///
	sComplex get();

	///
	Complex conjg();

	///
	double Cabs();

	///
	Complex Csqrt();

	///
	void add(const Complex& a, const Complex& b);

	///
	void sub(const Complex& a, const Complex& b);

	///
	void mul(const Complex& a, const Complex& b);

	///
	void div(const Complex& a, const Complex& b);

	///
	void mul(double x, const Complex& a);
};

}

#endif
