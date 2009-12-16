#include "torch5spro.h"
#include <cassert>

using namespace Torch;

///////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////
int main()
{

   	THRandom_manualSeed(950305);

	print("\n\nTesting 2D DCT\n\n");

	//
	int H = 8;
	int W = 8;
	FloatTensor image(H, W);

	for(int i = 0 ; i < H ; i++)
		for(int j = 0 ; j < W ; j++)
		{
			double random_ = (int) THRandom_uniform(0, 255);
	        	image(i,j) = random_;
		}

	image.print("x");

	//
	spDCT dct2d;
	print("Computing the DCT of x ...\n");
	dct2d.process(image);
	assert(dct2d.getNOutputs() == 1);
	dct2d.getOutput(0).print("F[x]");

	//
	spDCT idct2d(true);
	print("Computing the iDCT of F[x]...\n");
	idct2d.process(dct2d.getOutput(0));
	assert(idct2d.getNOutputs() == 1);
	idct2d.getOutput(0).print("inverse F[x]");

	return 0;

	//
	const int n_tests = 11;
	const int n_tests_N[n_tests] = { 2, 4, 8, 16, 32, 64, 128, 256, 1024, 2048, 4096};

	for(int t = 0 ; t < n_tests ; t++)
	{
		H = n_tests_N[t];

		for(int tt = 0 ; tt < n_tests ; tt++)
		{
			W = n_tests_N[tt];

			print("Testing %dx%d-points 2D DCT\n", H, W);

			FloatTensor x(H,W);

			for(int i = 0 ; i < H ; i++)
				for(int j = 0 ; j < W ; j++)
				{
					double random_ = THRandom_uniform(0, 255);
			        	x(i,j) = random_;
				}

			//
			spDCT dct2d;
			print("Computing the DCT of x ...\n");
			dct2d.process(x);

			//
			spDCT idct2d(true);
			print("Computing the iDCT of F[x]...\n");
			idct2d.process(dct2d.getOutput(0));

			const FloatTensor& out = (const FloatTensor&) idct2d.getOutput(0);

			double rmse = 0.0;
			for(int i = 0 ; i < H ; i++)
				for(int j = 0 ; j < W ; j++)
				{
			   		double z = x(i,j) - out.get(i,j);
					rmse += z*z;
				}
			rmse /= (double) (W*H);

			print("RMSE = %g\n\n", rmse);
		}

	}


	//
	print("\nOK\n");

	return 0;
}

