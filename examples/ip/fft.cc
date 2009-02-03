#include "ipFFT.h"
#include "Tensor.h"
#include <cassert>

using namespace Torch;

///////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////
int main()
{

   	THRandom_manualSeed(950305);

	print("\n\nTesting 1D FFT\n\n");

	//
	int N = 8;
	FloatTensor data(N);

	for(int i = 0 ; i < N ; i++)
	{
		double random_ = THRandom_uniform(0, 255);
	        data(i) = random_;
	}

	data.print("x");
	
	//
	ipFFT fft1d;
	print("Computing the FFT of x ...\n");
	fft1d.process(data);
	assert(fft1d.getNOutputs() == 1);
	fft1d.getOutput(0).print("F[x]");

	//
	ipFFT ifft1d(true);
	print("Computing the iFFT of F[x]...\n");
	ifft1d.process(fft1d.getOutput(0));
	assert(ifft1d.getNOutputs() == 1);
	ifft1d.getOutput(0).print("inverse F[x]");



	print("\n\nTesting 2D FFT\n\n");

	//
	int H = 4;
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
	ipFFT fft2d;
	print("Computing the FFT of x ...\n");
	fft2d.process(image);
	assert(fft2d.getNOutputs() == 1);
	fft2d.getOutput(0).print("F[x]");

	//
	ipFFT ifft2d(true);
	print("Computing the iFFT of F[x]...\n");
	ifft2d.process(fft2d.getOutput(0));
	assert(ifft2d.getNOutputs() == 1);
	ifft2d.getOutput(0).print("inverse F[x]");

	print("\nOK\n");

	return 0;
}

