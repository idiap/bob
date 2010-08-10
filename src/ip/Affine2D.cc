#include "ip/Affine2D.h"

namespace Torch {
	
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
