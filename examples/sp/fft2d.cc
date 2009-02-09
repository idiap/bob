#include "spFFT.h"
#include "Tensor.h"
#include <cassert>

using namespace Torch;

///////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////
int main()
{

   	THRandom_manualSeed(950305);

	print("\n\nTesting 2D FFT\n\n");

	//
	int H = 8;
	int W = 4;
	FloatTensor image(H, W);

	for(int i = 0 ; i < H ; i++)
		for(int j = 0 ; j < W ; j++)
		{
			double random_ = THRandom_uniform(0, 255);
	        	image(i,j) = random_;
		}

	image.print("x");
	
	//
	spFFT fft2d;
	print("Computing the FFT of x ...\n");
	fft2d.process(image);
	assert(fft2d.getNOutputs() == 1);
	fft2d.getOutput(0).print("F[x]");

	//
	spFFT ifft2d(true);
	print("Computing the iFFT of F[x]...\n");
	ifft2d.process(fft2d.getOutput(0));
	assert(ifft2d.getNOutputs() == 1);
	ifft2d.getOutput(0).print("inverse F[x]");

	//
	const int n_tests = 11;
	const int n_tests_N[n_tests] = { 2, 4, 8, 16, 32, 64, 128, 256, 1024, 2048, 4096};

	for(int t = 0 ; t < n_tests ; t++)
	{
		//
		for(int tt = 0 ; tt < n_tests ; tt++)
		{
			print("Testing %dx%d 2D FFT\n", n_tests_N[t], n_tests_N[tt]);

			//
			int H = n_tests_N[t];
			int W = n_tests_N[tt];
			FloatTensor image(H, W);

			for(int i = 0 ; i < H ; i++)
				for(int j = 0 ; j < W ; j++)
				{
					double random_ = THRandom_uniform(0, 255);
			        	image(i,j) = random_;
				}

			//image.print("x");
			
			//
			spFFT fft2d;
			print("  Computing the FFT of x ...\n");
			fft2d.process(image);
			//fft2d.getOutput(0).print("F[x]");

			//
			spFFT ifft2d(true);
			print("  Computing the iFFT of F[x]...\n");
			ifft2d.process(fft2d.getOutput(0));
			//ifft2d.getOutput(0).print("inverse F[x]");
			
			const FloatTensor& out = (const FloatTensor&) ifft2d.getOutput(0);

			double rmse = 0.0;
			for(int h = 0 ; h < H ; h++)
				for(int w = 0 ; w < W ; w++)
				{
			   		double z = image(h,w) - out.get(h,w);
					rmse += z*z;
				}
			rmse /= (double) (H*W);

			print("RMSE = %g\n\n", rmse);
		}
	}

	//
	print("\nOK\n");

	return 0;
}

