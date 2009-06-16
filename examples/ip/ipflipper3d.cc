#include "torch5spro.h"

using namespace Torch;

///////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////
int main()
{
	xtprobeImageFile xtprobe;
	Image image(1, 1, 3);

	// Load the image to play with
	const char* imagefilename = "../data/images/1001_f_g1_s01_1001_en_1.jpeg";
	CHECK_FATAL(xtprobe.load(image, imagefilename) == true);
	print("Loaded image of size [%dx%d] with [%d] planes.\n\n",
		image.getWidth(), image.getHeight(), image.getNPlanes());

	ipFlip flipper;

	// Flip the image
	print("flip\n");
	CHECK_FATAL(flipper.setBOption("vertical", true) == true);
	CHECK_FATAL(flipper.process(image) == true);
	CHECK_FATAL(flipper.getNOutputs() == 1);

	// Save it to some file
	Image flipImg(image.getWidth(), image.getHeight(), image.getNPlanes());
	CHECK_FATAL(flipImg.copyFrom(flipper.getOutput(0)) == true);

	const char* file_out = "1001_f_g1_s01_1001_en_1-vert.jpeg";
	CHECK_FATAL(xtprobe.save(flipImg, file_out) == true);

	// Flip the image over hor
	print("flip\n");
	CHECK_FATAL(flipper.setBOption("vertical", false) == true);
	CHECK_FATAL(flipper.process(image) == true);
	CHECK_FATAL(flipper.getNOutputs() == 1);

	// Save it to some file
	//Image flipImg(image.getWidth(), image.getHeight(), image.getNPlanes());
	CHECK_FATAL(flipImg.copyFrom(flipper.getOutput(0)) == true);

	const char *file_out2 = "1001_f_g1_s01_1001_en_1-hori.jpeg";
	CHECK_FATAL(xtprobe.save(flipImg, file_out2) == true);

	print("\nOK\n");

	return 0;
}

