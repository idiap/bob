#include "torch5spro.h"

using namespace Torch;

///////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////
int main()
{
	xtprobeImageFile xtprobe;
	Image image(1, 1, 3);
	Image crop_image;

	// Load the image to play with
	const char* imagefilename = "../data/images/1001_f_g1_s01_1001_en_1.jpeg";
	CHECK_FATAL(xtprobe.load(image, imagefilename) == true);
	print("Loaded image of size [%dx%d] with [%d] planes.\n\n",
		image.getWidth(), image.getHeight(), image.getNPlanes());

	ipCrop cropper;

	// Crop some random parts
	const int n_tests = 3;
	srand((unsigned int)time(0));

	for (int t = 0; t < n_tests; t ++)
	{
		const int crop_x = rand() % image.getWidth();
		const int crop_y = rand() % image.getHeight();
		const int crop_w = (image.getWidth() - crop_x) / 2;
		const int crop_h = (image.getHeight() - crop_y) / 2;

		// Crop the area
		print("[%d/%d]: cropping [%d, %d, %d, %d] area ...\n",
			t + 1, n_tests, crop_x, crop_y, crop_w, crop_h);
		CHECK_FATAL(cropper.setIOption("x", crop_x) == true);
		CHECK_FATAL(cropper.setIOption("y", crop_y) == true);
		CHECK_FATAL(cropper.setIOption("w", crop_w) == true);
		CHECK_FATAL(cropper.setIOption("h", crop_h) == true);
		CHECK_FATAL(cropper.process(image) == true);
		CHECK_FATAL(cropper.getNOutputs() == 1);

		// Save it to some file
		crop_image.resize(crop_w, crop_h, image.getNPlanes());
		CHECK_FATAL(crop_image.getWidth() == crop_w);
		CHECK_FATAL(crop_image.getHeight() == crop_h);
		CHECK_FATAL(crop_image.copyFrom(cropper.getOutput(0)) == true);

		char str[200];
		sprintf(str, "crop_%d_%d_%d_%d.jpg", crop_x, crop_y, crop_w, crop_h);
		CHECK_FATAL(xtprobe.save(crop_image, str) == true);
	}

	print("\nOK\n");

	return 0;
}

