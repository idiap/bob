#include "ipScaleYX.h"
#include "Image.h"
#include "xtprobeImageFile.h"

using namespace Torch;

///////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
	xtprobeImageFile xtprobe;
	Image image(1, 1, 3);
	Image scale_image;

	// Load the image to play with
	const char* imagefilename = "../data/images/1001_f_g1_s01_1001_en_1.jpeg";
	CHECK_FATAL(xtprobe.open(imagefilename, "r") == true);
	CHECK_FATAL(image.loadImage(xtprobe) == true);
	xtprobe.close();
	print("Loaded image of size [%dx%d] with [%d] planes.\n\n",
		image.getWidth(), image.getHeight(), image.getNPlanes());

	ipScaleYX scaler;

	// Crop some random parts
	const int n_tests = 10;
	srand((unsigned int)time(0));

	for (int t = 0; t < n_tests; t ++)
	{
		const int scale_w = 1 + rand() % (image.getWidth() * 2);
		const int scale_h = 1 + rand() % (image.getHeight() * 2);

		// Scale to this size
		print("[%d/%d]: scaling [%d x %d] => [%d, %d] ...\n",
			t + 1, n_tests, image.getWidth(), image.getHeight(), scale_w, scale_h);
		CHECK_FATAL(scaler.setIOption("width", scale_w) == true);
		CHECK_FATAL(scaler.setIOption("height", scale_h) == true);
		CHECK_FATAL(scaler.process(image) == true);
		CHECK_FATAL(scaler.getNOutputs() == 1);

		// Save it to some file
		scale_image.resize(scale_w, scale_h, image.getNPlanes());
		CHECK_FATAL(scale_image.getWidth() == scale_w);
		CHECK_FATAL(scale_image.getHeight() == scale_h);
		CHECK_FATAL(scale_image.copyFrom(scaler.getOutput(0)) == true);

		char str[200];
		sprintf(str, "scale_%d_%d.jpg", scale_w, scale_h);
		CHECK_FATAL(xtprobe.open(str, "w") == true);
		CHECK_FATAL(scale_image.saveImage(xtprobe) == true);
		xtprobe.close();
	}

	print("\nOK\n");

	return 0;
}

