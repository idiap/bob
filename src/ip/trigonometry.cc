#include "ip/trigonometry.h"

namespace Torch
{

double degree2radian(double d)
{
	return d * M_PI / 180.0;
}

double radian2degree(double r)
{
	return r * 180.0 / M_PI;
}

}
