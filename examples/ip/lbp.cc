#include "ipLBP4R.h"
#include "ipLBP8R.h"
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
	Image image(1, 1, 1);
	Image lbp_image;

	// Load the image to play with
	const char* imagefilename = "../data/images/003_1_1.pgm";
	assert(xtprobe.open(imagefilename, "r") == true);
	assert(image.loadImage(xtprobe) == true);
	xtprobe.close();
	print("Loaded image of size [%dx%d] with [%d] planes.\n\n",
		image.getWidth(), image.getHeight(), image.getNPlanes());

	// Build the <ipLBP> objects to play with
	const int n_ip_lbps = 12;
	const char* str_ip_lbps[n_ip_lbps] =
		{
			"4R", "4R_avg", "4R_avg_add_bit",
			"4R_ri", "4R_avg_u2", "4R_u2ri",

			"8R", "8R_avg", "8R_avg_add_bit",
			"8R_ri", "8R_u2", "8R_u2ri",
		};

	// Run each LBP type
	for (int i = 0; i < n_ip_lbps; i ++)
	{
		ipLBP* ip_lbp = (i < 6) ? ((ipLBP*)new ipLBP4R(1)) : ((ipLBP*)new ipLBP8R(1));

		// Initialize the <ipLBP>
		const int w = image.getWidth();
		const int h = image.getHeight();
		assert(ip_lbp->setInputSize(w, h) == true);

		// Set the parameters for this <ipLBP>
		const bool param_avg = i == 1 || i == 2 || i == 7 || i == 8;
		const bool param_avg_add_bit = i == 2 || i == 8;
		const bool param_uniform = i == 4 || i == 5 || i == 10 || i == 11;
		const bool param_rot_invariant = i == 3 || i == 5 || i == 9 || i == 11;

		assert(ip_lbp->setBOption("ToAverage", param_avg) == true);
		assert(ip_lbp->setBOption("AddAvgBit", param_avg_add_bit) == true);
		assert(ip_lbp->setBOption("Uniform", param_uniform) == true);
		assert(ip_lbp->setBOption("RotInvariant", param_rot_invariant) == true);

		const int max_lbp = ip_lbp->getMaxLabel();
		const float inv_max_lbp = 255.0f / (max_lbp + 0.0f);

		// Debug message
		print("Test [%d/%d]: avg = [%d], avg_add_bit = [%d], uniform = [%d], rot_invariant = [%d] => max_lbp = %d\n",
			i + 1, n_ip_lbps,
			param_avg, param_avg_add_bit, param_uniform, param_rot_invariant,
			max_lbp);

		// Build the lbp image (scale each LBP code to the maximum possible value)
		assert(lbp_image.resize(image.getWidth(), image.getHeight(), 1) == true);
		for (int x = 1; x < w - 1; x ++)
			for (int y = 1; y < h - 1; y ++)
			{
				assert(ip_lbp->setXY(x, y) == true);
				assert(ip_lbp->process(image) == true);
				assert(ip_lbp->getLBP() >= 0);
				assert(ip_lbp->getLBP() < max_lbp);
				lbp_image.set(y, x, 0, (short)(inv_max_lbp * ip_lbp->getLBP() + 0.5f));
			}

		// Save the resulted image
		char str[200];
		//sprintf(str, "%s_%s.jpg", imagefilename, str_ip_lbps[i]);
		sprintf(str, "image_%s.jpg", str_ip_lbps[i]);
		assert(xtprobe.open(str, "w") == true);
		assert(lbp_image.saveImage(xtprobe) == true);
		xtprobe.close();

		delete ip_lbp;
	}

	print("\nOK\n");

	return 0;
}

