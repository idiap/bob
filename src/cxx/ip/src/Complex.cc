/**
 * @file cxx/ip/src/Complex.cc
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
#include <math.h>

#include "ip/Complex.h"

namespace Torch {

sComplex Complex::get()
{
	sComplex c;

	c.r = r;
	c.i = i;

	return c;
}

Complex Complex::conjg()
{
	Complex c(r, -i);

	return c;
}

double Complex::Cabs()
{
	double x, y , ans, temp;

	x = fabs(r);
	y = fabs(i);

	if(x == 0.0) return y;
	else if(y == 0.0) return x;
	else if(x > y)
	{
		temp = y / x;
		ans = x * sqrt(1.0 + temp*temp);
	}
	else
	{
		temp = x / y;
		ans = y * sqrt(1.0 + temp*temp);
	}
	return ans;
}

Complex Complex::Csqrt()
{
	Complex c;

	double x, y, w, z_;

	if ((r == 0.0) && (i == 0.0))
	{
		c.r=0.0;
		c.i=0.0;
	}
	else
	{
		x = fabs(r);
		y = fabs(i);

		if(x >= y)
		{
			z_ = y / x;
			w = sqrt(x) * sqrt(0.5*(1.0 + sqrt(1.0 + z_*z_)));
		}
		else
		{
			z_ = x / y;
			w = sqrt(y) * sqrt(0.5*(z_ + sqrt(1.0 + z_*z_)));
		}

		if (r >= 0.0)
		{
			c.r = w;
			c.i = i / (2.0*w);
		}
		else
		{
			c.i = (i >= 0) ? w : -w;
			c.r = i/(2.0*c.i);
		}
	}

	return c;
}

void Complex::add(const Complex& a, const Complex& b)
{
	r = a.r + b.r;
	i = a.i + b.i;
}

void Complex::sub(const Complex& a, const Complex& b)
{
	r = a.r - b.r;
	i = a.i - b.i;
}


void Complex::mul(const Complex& a, const Complex& b)
{
	r = a.r * b.r - a.i * b.i;
	i = a.i * b.r + a.r * b.i;
}

void Complex::div(const Complex& a, const Complex& b)
{
	double z, den;

	if(fabs(b.r) >= fabs(b.i))
	{
		z = b.i / b.r;
		den = b.r + z * b.i;
		r = (a.r + z * a.i) / den;
		i = (a.i - z * a.r) / den;
	}
	else
	{
		z = b.r / b.i;
		den = b.i + z * b.r;
		r = (a.r * z + a.i) / den;
		i = (a.i * z - a.r) / den;
	}
}

void Complex::mul(double x, const Complex& a)
{
	r = x * a.r;
	i = x * a.i;
}


}
