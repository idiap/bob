#include "ipSmoothGaussian.h"
#include "Image.h"
#include "xtprobeImageFile.h"
#include "CmdLine.h"

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

	CHECK_FATAL(xtprobe.load(image, image_filename) == true);
	print("Loaded image of size [%dx%d] with [%d] planes.\n\n",
		image.getWidth(), image.getHeight(), image.getNPlanes());

	// Set the Gaussian filter and smooth the image
	ipSmoothGaussian gaussian;
	CHECK_FATAL(gaussian.setIOption("RadiusX", radius_x) == true);
	CHECK_FATAL(gaussian.setIOption("RadiusY", radius_y) == true);
	CHECK_FATAL(gaussian.setDOption("Sigma", sigma) == true);
	CHECK_FATAL(gaussian.process(image) == true);
	CHECK_FATAL(gaussian.getNOutputs() == 1);
	CHECK_FATAL(gaussian.getOutput(0).getDatatype() == Tensor::Short);

	// Save the smoothed image
	CHECK_FATAL(image.copyFrom(gaussian.getOutput(0)) == true);
	CHECK_FATAL(xtprobe.save(image, "smoothed.jpg") == true);

	print("\nOK\n");

	return 0;
}

