#include "Ray2D.h"
#include "Color.h"

namespace Torch {

void Ray2D::saveFile(File *file)
{
   	P0.saveFile(file);
   	P1.saveFile(file);
}

void Ray2D::loadFile(File *file)
{
   	P0.loadFile(file);
   	P1.loadFile(file);
}

const char *Ray2D::sprint()
{
	sprintf(buf_sprint, "[(%g, %g) (%g, %g))", P0.x, P0.y, P1.x, P1.y);

	return buf_sprint;
}

void Ray2D::draw(Image *image, Color color)
{
   	// Warning: this is not exactly correct as a ray is semi-finite
	// TODO: computes intersections of the line with the image and draw it

	image->drawLine((int)P0.x, (int)P0.y, (int)P1.x, (int)P1.y, color);
}

}
