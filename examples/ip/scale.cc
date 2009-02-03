#include "ipScaleYX.h"
#include "Image.h"
#include "xtprobeImageFile.h"
#include <cassert>

using namespace Torch;

///////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////
int main()
{
	xtprobeImageFile xtprobe;
	Image image(1, 1, 3);
	Image scale_image;

	// Load the image to play with
	const char* imagefilename = "../data/images/1001_f_g1_s01_1001_en_1.jpeg";
	assert(xtprobe.open(imagefilename, "r") == true);
	assert(image.loadImage(xtprobe) == true);
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
		assert(scaler.setOutputSize(scale_w, scale_h) == true);
		assert(scaler.process(image) == true);
		assert(scaler.getNOutputs() == 1);

		// Save it to some file
		scale_image.resize(scale_w, scale_h, image.getNPlanes());
		assert(scale_image.getWidth() == scale_w);
		assert(scale_image.getHeight() == scale_h);
		assert(scale_image.copyFrom(scaler.getOutput(0)) == true);

		char str[200];
		sprintf(str, "scale_%d_%d.jpg", scale_w, scale_h);
		assert(xtprobe.open(str, "w") == true);
		assert(scale_image.saveImage(xtprobe) == true);
		xtprobe.close();
	}

	print("\nOK\n");

	return 0;
}

