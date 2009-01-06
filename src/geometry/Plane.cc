#include "Plane.h"
#include "Color.h"

namespace Torch {

void Plane::saveFile(File *file)
{
   	P0.saveFile(file);
   	n.saveFile(file);
}

void Plane::loadFile(File *file)
{
   	P0.loadFile(file);
   	n.loadFile(file);
}

const char *Plane::sprint()
{
	sprintf(buf_sprint, "{(%g, %g, %g) (%g, %g, %g)}", P0.x, P0.y, P0.z, n.x, n.y, n.z);

	return buf_sprint;
}

void Plane::draw(Image *image, Color color)
{
   	message("Don't know how to draw in 3D.");
}

}
