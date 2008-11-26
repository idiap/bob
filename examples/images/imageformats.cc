#include "Image.h"
#include "xtprobeImageFile.h"
#include "pgmImageFile.h"
#include "ppmImageFile.h"
#include "gifImageFile.h"
#include "jpegImageFile.h"
#include "tiffImageFile.h"
#include "Color.h"
#include <cassert>

using namespace Torch;

///////////////////////////////////////////////////////////////////////////

enum ImageType
{
	ImageType_pgm,
	ImageType_ppm,
	ImageType_gif,
	ImageType_jpeg,
	ImageType_tiff
};

const char* ImageTypeExt[] =
{
	"pgm",
	"ppm",
	"gif",
	"jpeg",
	"tiff"
};

///////////////////////////////////////////////////////////////////////////
// Returns an ImageFile loader object accordingly with the given ImageType
///////////////////////////////////////////////////////////////////////////

ImageFile* BuildImageFile(ImageType type)
{
	switch(type)
	{
	case ImageType_pgm:
		return new pgmImageFile();

	case ImageType_ppm:
		return new ppmImageFile();

	case ImageType_gif:
		return new gifImageFile();

	case ImageType_tiff:
		return new tiffImageFile();

	case ImageType_jpeg:
	default:
		return new jpegImageFile();
	}
}

///////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////
int main()
{
	// Prepare image dataset to play with
	const int n_images = 5;
	const char imagefiles[n_images][64] =
		{
			"../data/images/003_1_1.pgm",
			"../data/images/caution.gif",
			"../data/images/gloves.ppm",
			"../data/images/9001_f_wm_s01_9001_en_1.ppm",
			"../data/images/1001_f_g1_s01_1001_en_1.jpeg"
		};
	const ImageType input_types[n_images] =
		{
			ImageType_pgm,
			ImageType_gif,
			ImageType_ppm,
			ImageType_ppm,
			ImageType_jpeg
		};
	const ImageType output_types[n_images] =
		{
			ImageType_pgm,
			ImageType_ppm,
			ImageType_pgm,
			ImageType_ppm,
			ImageType_jpeg
		};

	// Create a dummy 3 color channel image (it will be resized automatically)
	Image image(10, 10, 3);

	const int n_tests = 100;
	const int n_draws = 8;

	// Do the tests ...
	srand((unsigned int)time(0));
	for (int t = 0; t < n_tests; t ++)
	{
		char filename[200];

		const int index_input = rand() % n_images;
		const int index_output = rand() % n_images;
		const ImageType input_type = input_types[index_input];
		const ImageType output_type = output_types[index_output];

		// Load the image
		ImageFile* iloader = BuildImageFile(input_type);
		assert(iloader->open(imagefiles[index_input], "r") == true);
		assert(image.loadImage(*iloader) == true);
		delete iloader;

		// Draw something on the image
		for (int i = 0; i < n_draws; i ++)
		{
			image.drawLine(	rand() % image.getWidth(), rand() % image.getHeight(),
					rand() % image.getWidth(), rand() % image.getHeight(),
					Torch::red);
			image.drawPixel(rand() % image.getWidth(), rand() % image.getHeight(),
					Torch::blue);
		}

		// Save the image to a random format
		ImageFile* isaver = BuildImageFile(output_type);
		sprintf(filename, "image.%d.%s", t + 1, ImageTypeExt[output_type]);
		assert(isaver->open(filename, "w") == true);
		assert(image.saveImage(*isaver) == true);
		delete isaver;

		print("\txxxImageFile PASSED: [%d/%d] \r", t + 1, n_tests);
	}

	print("\n");

	// Test the xtprobeImageFile
	xtprobeImageFile xtprobe;
	for (int t = 0; t < n_tests; t ++)
	{
		assert(xtprobe.open(imagefiles[t % n_images], "r") == true);
		assert(image.loadImage(xtprobe) == true);
		xtprobe.close();

		print("\txtprobeImageFile PASSED: [%d/%d] \r", t + 1, n_tests);
	}

	print("\nOK\n");

	return 0;
}

