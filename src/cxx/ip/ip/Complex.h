/**
 * @file cxx/ip/ip/Complex.h
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
#ifndef COMPLEX_INC
#define COMPLEX_INC

#include "ip/vision.h"

namespace bob {

/** This class is designed to handle complex numbers

    @author Sebastien Marcel (marcel@idiap.ch)
    @version 2.0
    \date
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
