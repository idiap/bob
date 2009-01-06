#include "Triangle2D.h"
#include "Color.h"

namespace Torch {

void Triangle2D::saveFile(File *file)
{
   	P0.saveFile(file);
   	P1.saveFile(file);
   	P2.saveFile(file);
}

void Triangle2D::loadFile(File *file)
{
   	P0.loadFile(file);
   	P1.loadFile(file);
   	P2.loadFile(file);
}

const char *Triangle2D::sprint()
{
	sprintf(buf_sprint, "[(%g, %g) (%g, %g) (%g, %g)]",
	      		P0.x, P0.y,
			P1.x, P1.y,
			P2.x, P2.y);

	return buf_sprint;
}

void Triangle2D::draw(Image *image, Color color)
{
	image->drawLine((int)P0.x, (int)P0.y, (int)P1.x, (int)P1.y, color);
	image->drawLine((int)P1.x, (int)P1.y, (int)P2.x, (int)P2.y, color);
	image->drawLine((int)P2.x, (int)P2.y, (int)P0.x, (int)P0.y, color);
}

}
