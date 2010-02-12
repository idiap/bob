#include "torch5spro.h"

using namespace Torch;

///////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
	char* image_filename = 0;

	// Read the command line
	CmdLine cmd;
	cmd.info("Testing program for Histogram Equalization.\n");
	cmd.addSCmdArg("image", &image_filename, "input image");

	cmd.read(argc, argv);

	// Load the image
	Image image_(1, 1, 3);
	xtprobeImageFile xtprobe;
	CHECK_FATAL(xtprobe.load(image_, image_filename) == true);

	print("Processing image [width = %d, height = %d, nplanes = %d] ...\n",
		image_.size(1), image_.size(0), image_.size(2));

        // Convert it in Gray level (1 channel)
        Image image(image_.size(1),image_.size(0),1);
        CHECK_FATAL(image.copyFrom(image_) == true); 


	Image histoEqual_image(1, 1, image.size(2));
	ipHistoEqual histoEqual;

	// Perform Histogram Equalization
	print("Performing Histogram Equalization...\n");
	CHECK_FATAL(histoEqual.process(image) == true);
	CHECK_FATAL(histoEqual.getNOutputs() == 1);
	CHECK_FATAL(histoEqual.getOutput(0).getDatatype() == Tensor::Short);
	const ShortTensor& output = (const ShortTensor&)histoEqual.getOutput(0);


	// Save the shifted image
	char str[256];
	sprintf(str, "histoEqual.pgm");
	CHECK_FATAL(histoEqual_image.resize(output.size(1), output.size(0), output.size(2)) == true);
	CHECK_FATAL(histoEqual_image.copyFrom(output) == true);
	CHECK_FATAL(xtprobe.save(histoEqual_image, str) == true);


	return 0;
}

