#include "ipIntegral.h"
#include "Tensor.h"
#include <cassert>

using namespace Torch;

//////////////////////////////////////////////////////////////////////////////////
// Generate a tensor 2D/3D tensor of random values, given its type and size
//////////////////////////////////////////////////////////////////////////////////

Tensor* generateCharTensor(int width, int height, int nplanes)
{
        if (nplanes < 1)        // 2D
        {
                CharTensor* tensor = new CharTensor(height, width);
                for (int j = 0; j < height; j ++)
                        for (int i = 0; i < width; i ++)
                        {
                                tensor->set(j, i, rand() % 125);
                        }
                return tensor;
        }
        else                    // 3D
        {
                CharTensor* tensor = new CharTensor(height, width, nplanes);
                for (int j = 0; j < height; j ++)
                        for (int i = 0; i < width; i ++)
                                for (int p = 0; p < nplanes; p ++)
                                {
                                        tensor->set(j, i, p, rand() % 125);
                                }
                return tensor;
        }
}

Tensor* generateShortTensor(int width, int height, int nplanes)
{
        if (nplanes < 1)        // 2D
        {
                ShortTensor* tensor = new ShortTensor(height, width);
                for (int j = 0; j < height; j ++)
                        for (int i = 0; i < width; i ++)
                        {
                                tensor->set(j, i, rand() % 256);
                        }
                return tensor;
        }
        else                    // 3D
        {
                ShortTensor* tensor = new ShortTensor(height, width, nplanes);
                for (int j = 0; j < height; j ++)
                        for (int i = 0; i < width; i ++)
                                for (int p = 0; p < nplanes; p ++)
                                {
                                        tensor->set(j, i, p, rand() % 256);
                                }
                return tensor;
        }
}

Tensor* generateFloatTensor(int width, int height, int nplanes)
{
        if (nplanes < 1)        // 2D
        {
                FloatTensor* tensor = new FloatTensor(height, width);
                for (int j = 0; j < height; j ++)
                        for (int i = 0; i < width; i ++)
                        {
                                tensor->set(j, i, (rand() % 256) / 256.0f);
                        }
                return tensor;
        }
        else                    // 3D
        {
                FloatTensor* tensor = new FloatTensor(height, width, nplanes);
                for (int j = 0; j < height; j ++)
                        for (int i = 0; i < width; i ++)
                                for (int p = 0; p < nplanes; p ++)
                                {
                                        tensor->set(j, i, p, (rand() % 256) / 256.0f);
                                }
                return tensor;
        }
}

//////////////////////////////////////////////////////////////////////////////////
// MAIN
//////////////////////////////////////////////////////////////////////////////////

int main()
{
        srand((unsigned int)time(0));

	const int n_full_tests = 5;
	const int n_tests[n_full_tests] = { 1, 100, 100, 100, 100 };
	const int widths[n_full_tests] = { 4, 64, 128, 256, 512 };
	const int heights[n_full_tests] = { 3, 80, 160, 320, 640 };

	// We'll reuse the <ipIntegral> for any kind of 2D/3D tensor!!!
	ipIntegral ipii;

	// Do the set of tests
	for (int n = 0; n < n_full_tests; n ++)
	{
		const int w = widths[n];
		const int h = heights[n];

		// Generate a random tensor of this size
		Tensor* input = 0;
		if (n == 0)
		{
		        input = generateCharTensor(w, h, -1);   // 2D;
		        input->print("INPUT-----------------");
		}
		else
		{
		        switch (rand() % 3)
		        {
                        case 0:
                                input = generateCharTensor(w, h, rand() % 2 == 0 ? -1 : 3);// 2D/3D
                                break;
                        case 1:
                                input = generateShortTensor(w, h, rand() % 2 == 0 ? -1 : 3);// 2D/3D
                                break;
                        case 2:
                        default:
                                input = generateFloatTensor(w, h, rand() % 2 == 0 ? -1 : 3);// 2D/3D
                                break;
		        }
		}

		print("Testing with 2D/3D tensors of [w = %d, h = %d], with [%d] planes\n",
                        w, h, input->nDimension() == 3 ? input->size(2) : -1);

		// Compute its integral image multiple times
		const int ntests = n_tests[n];
		for (int t = 0; t < ntests; t ++)
		{
			assert(ipii.process(*input) == true);
			assert(ipii.getNOutputs() == 1);

			if (n == 0)
			{
				ipii.getOutput(0).print("INTEGRAL-----------------");
			}
			print("\tPASSED: [%d/%d]\r", t + 1, ntests);
		}
		print("\n");

                // Cleanup
		delete input;
	}

	print("\nOK\n");

	return 0;
}

