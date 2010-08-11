#include "ip/Ray2D.h"
#include "ip/Color.h"

namespace Torch {

void Ray2D::draw(Image *image, Color color)
{
   	// Warning: this is not exactly correct as a ray is semi-finite
	// TODO: computes intersections of the line with the image and draw it

	image->drawLine((int)P0.get(0), (int)P0.get(1), (int)P1.get(0), (int)P1.get(1), color);
}

}
