#include "ipShift.h"
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

	// Read the command line
	CmdLine cmd;
	cmd.info("Testing program for shifting images.\n");
	cmd.addSCmdArg("image", &image_filename, "input image");

	cmd.read(argc, argv);

	// Load the image
	Image image(1, 1, 3);
	xtprobeImageFile xtprobe;
	CHECK_FATAL(xtprobe.open(image_filename, "r") == true);
	CHECK_FATAL(image.loadImage(xtprobe) == true);
	xtprobe.close();

	print("Processing image [width = %d, height = %d, nplanes = %d] ...\n",
		image.size(1), image.size(0), image.size(2));

	Image shifted_image(image.size(1), image.size(0), image.size(2));
	ipShift shifter;

	// Do some random shifts
	const int n_tests = 10;
	srand((unsigned int)time(0));
	for (int i = 0; i < n_tests; i ++)
	{
		const int dx = rand() % (image.size(1) / 4) - (image.size(1) / 8);
		const int dy = rand() % (image.size(0) / 4) - (image.size(0) / 8);

		print("[%d/%d]: shifting with [%d - %d]...\n", i + 1, n_tests, dx, dy);
		CHECK_FATAL(shifter.setIOption("shiftx", dx) == true);
		CHECK_FATAL(shifter.setIOption("shifty", dy) == true);
		CHECK_FATAL(shifter.process(image) == true);
		CHECK_FATAL(shifter.getNOutputs() == 1);
		CHECK_FATAL(shifter.getOutput(0).getDatatype() == Tensor::Short);

		// Save the shifted image
		char str[256];
		sprintf(str, "shift_%d_%d.jpg", dx, dy);
		CHECK_FATAL(shifted_image.copyFrom(shifter.getOutput(0)) == true);
		CHECK_FATAL(xtprobe.open(str, "w") == true);
		CHECK_FATAL(shifted_image.saveImage(xtprobe) == true);
		xtprobe.close();
	}

	print("\nOK\n");

	return 0;
}

