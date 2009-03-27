#include "ShapeModel.h"
#include "Circle2D.h"
#include "Line2D.h"

#include "Image.h"
#include "Color.h"
//#include "ipRotate.h"

namespace Torch {

ShapeModel::ShapeModel(int n_bbx_points_)
{
	if(n_bbx_points_ <= 0) error("ShapeModel::ShapeModel() You should give a number of bbx points >= 0");

   	//
	n_ldm_points = 0;
	ldm_points = NULL;

	//
	n_bbx_points = n_bbx_points_;
	bbx_points = new sPoint2D[n_bbx_points];
	for(int i = 0 ; i < n_bbx_points ; i++)
		 bbx_points[i].x = bbx_points[i].y = 0.0;
}

/*
void ShapeModel::rotate(ipRotate *rot)
{
	sPoint2D tmp_;

	for(int i = 0 ; i < n_ldm_points ; i++)
	{
		rot->rotate(ldm_points[i].x, ldm_points[i].y, &tmp_.x, &tmp_.y);
		ldm_points[i] = tmp_;
	}
}
*/

void ShapeModel::drawLDM(Image *image)
{
	Circle2D circle;

	for(int i = 0 ; i < n_ldm_points ; i++)
	{
		circle.reset(ldm_points[i], 4);
		circle.draw(image, green);
	}
}

void ShapeModel::drawBBX(Image *image)
{
	//Line2D line;

   	warning("ShapeModel::drawBBX() not implemented");

	//line.reset(bbx_points[0], bbx_points[1]);
	//line.draw(image, yellow);
}

ShapeModel::~ShapeModel()
{
	delete[] bbx_points;
	delete[] ldm_points;
}

}
