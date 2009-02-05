#include "ipRotate.h"
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

	// Read the command line
	CmdLine cmd;
	cmd.info("Testing program for rotating images.\n");
	cmd.addSCmdArg("image", &image_filename, "input image");

	cmd.read(argc, argv);

	// Load the image
	Image image(1, 1, 3);
	xtprobeImageFile xtprobe;
	assert(xtprobe.open(image_filename, "r") == true);
	assert(image.loadImage(xtprobe) == true);
	xtprobe.close();

	print("Processing image [width = %d, height = %d, nplanes = %d] ...\n",
		image.size(1), image.size(0), image.size(2));

	Image rotated_image(1, 1, image.size(2));
	ipRotate rotator;

	// Rotate the image with a delta around a random position near center
	srand((unsigned int)time(0));
	const double dangle = 5.0;
	double angle = 0.0;
	while (angle < 365.0)
	{
		const int cx = image.size(1) / 2 + rand() % (image.size(1) / 16) - (image.size(1) / 32);
		const int cy = image.size(0) / 2 + rand() % (image.size(0) / 16) - (image.size(0) / 32);

		print("Rotating [%g] degrees with the center [%d, %d]...\n", angle, cx, cy);
		assert(rotator.setIOption("centerx", cx) == true);
		assert(rotator.setIOption("centery", cy) == true);
		assert(rotator.setDOption("angle", angle) == true);
		assert(rotator.process(image) == true);
		assert(rotator.getNOutputs() == 1);
		assert(rotator.getOutput(0).getDatatype() == Tensor::Short);
		const ShortTensor& output = (const ShortTensor&)rotator.getOutput(0);

		// Save the shifted image
		char str[256];
		sprintf(str, "rotate_%g_%d_%d.jpg", angle, cx, cy);
		assert(rotated_image.resize(output.size(1), output.size(0), output.size(2)) == true);
		assert(rotated_image.copyFrom(output) == true);
		assert(xtprobe.open(str, "w") == true);
		assert(rotated_image.saveImage(xtprobe) == true);
		xtprobe.close();

		angle += dangle;
	}

	print("\nOK\n");

	return 0;
}

