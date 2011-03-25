#include "ip/Triangle2D.h"
#include "ip/OldColor.h"

namespace Torch {

void Triangle2D::draw(Image *image, Color color)
{
	image->drawLine((int)P0.get(0), (int)P0.get(1), (int)P1.get(0), (int)P1.get(1), color);
	image->drawLine((int)P1.get(0), (int)P1.get(1), (int)P2.get(0), (int)P2.get(1), color);
	image->drawLine((int)P2.get(0), (int)P2.get(1), (int)P0.get(0), (int)P0.get(1), color);
}

}
