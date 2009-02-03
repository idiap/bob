#include "ipSmoothGaussian.h"
#include "Image.h"
#include "xtprobeImageFile.h"
#include "CmdLine.h"
#include <cassert>

using namespace Torch;

///////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
	char* image_filename = 0;
	int radius_x = 1;
	int radius_y = 1;
	double sigma = 0.25;

	// Read the command line
	CmdLine cmd;
	cmd.info("Testing program for Gaussian smoothing.\n");
	cmd.addSCmdArg("image", &image_filename, "input image");
	cmd.addICmdOption("-radius_x", &radius_x, 1, "Gaussian kernel's Ox radius");
	cmd.addICmdOption("-radius_y", &radius_y, 1, "Gaussian kernel's Oy radius");
	cmd.addDCmdOption("-sigma", &sigma, 5.0, "Gaussian kernel's variance");

	cmd.read(argc, argv);

	// Load the image to play with
	xtprobeImageFile xtprobe;
	Image image(1, 1, 3);

	assert(xtprobe.open(image_filename, "r") == true);
	assert(image.loadImage(xtprobe) == true);
	xtprobe.close();
	print("Loaded image of size [%dx%d] with [%d] planes.\n\n",
		image.getWidth(), image.getHeight(), image.getNPlanes());

	// Set the Gaussian filter and smooth the image
	ipSmoothGaussian gaussian;
	assert(gaussian.setIOption("RadiusX", radius_x) == true);
	assert(gaussian.setIOption("RadiusY", radius_y) == true);
	assert(gaussian.setDOption("Sigma", sigma) == true);
	assert(gaussian.process(image) == true);
	assert(gaussian.getNOutputs() == 1);
	assert(gaussian.getOutput(0).getDatatype() == Tensor::Short);

	// Save the smoothed image
	assert(image.copyFrom(gaussian.getOutput(0)) == true);
	assert(xtprobe.open("smoothed.jpg", "w+") == true);
	assert(image.saveImage(xtprobe) == true);
	xtprobe.close();

	print("\nOK\n");

	return 0;
}

