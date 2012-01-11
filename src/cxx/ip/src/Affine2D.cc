/**
 * @file cxx/ip/src/Affine2D.cc
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
#include "ip/Affine2D.h"

namespace bob {
	
Affine2D::Affine2D(Matrix2D &rs_, Vector2D &t_)
{ 
	rs = rs_;
	t = t_;
}

Point2D Affine2D::operator*(Point2D &p)
{
	Point2D p_;

	p_ = (p - t) * rs;
	
	return p_;
}

Vector2D Affine2D::operator*(Vector2D &v)
{
	Vector2D v_;

	v_ = (v - t) * rs;
	
	return v_;
}

Rectangle2D Affine2D::operator*(Rectangle2D &r)
{
	Rectangle2D r_;

	r_ = (r - t) * rs;
	
	return r_;
}

}
