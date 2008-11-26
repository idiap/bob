#include "ipIntegralImage.h"
#include "Image.h"
#include <cassert>

using namespace Torch;

int main()
{
	const int n_full_tests = 5;
	const int n_tests[n_full_tests] = { 1, 100, 100, 100, 100 };
	const int widths[n_full_tests] = { 4, 64, 128, 256, 512 };
	const int heights[n_full_tests] = { 3, 80, 160, 320, 640 };
	const int n_planes[n_full_tests] = { 3, 3, 3, 3, 3 };

	// we'll reuse these objects
	Image image;
	ipIntegralImage ipiimage;

	// Do the set of tests
	for (int n = 0; n < n_full_tests; n ++)
	{
		const int w = widths[n];
		const int h = heights[n];
		const int np = n_planes[n];

		print("Testing with image of: w = %d, h = %d, np = %d\n", w, h, np);

		assert(image.resize(w, h, np) == true);
		assert(ipiimage.setInputSize(w, h) == true);

		// Generate a random image of this size
		srand((unsigned int)time(0));
		for (int j = 0; j < h; j ++)
			for (int i = 0; i < w; i ++)
				for (int p = 0; p < np; p ++)
				{
					image.set(j, i, p, rand() % 256);
				}
		if (n == 0)
		{
			image.print("IMAGE-----------------");
		}

		// Compute its integral image multiple times
		const int ntests = n_tests[n];
		for (int t = 0; t < ntests; t ++)
		{
			assert(ipiimage.process(image) == true);
			assert(ipiimage.getNOutputs() == 1);

			const IntTensor& iimage = *((IntTensor*)&(ipiimage.getOutput(0)));
			if (n == 0)
			{
				iimage.print("IIMAGE-----------------");
			}
			print("\tPASSED: [%d/%d]\r", t + 1, ntests);
		}
		print("\n");
	}

	print("\nOK\n");

	return 0;
}

