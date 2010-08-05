#include "Segment2D.h"
#include "Color.h"

namespace Torch {

void Segment2D::draw(Image *image, Color color)
{
	image->drawLine((int)P0.get(0), (int)P0.get(1), (int)P1.get(0), (int)P1.get(1), color);
}

}
