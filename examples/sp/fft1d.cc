#include "torch5spro.h"
#include <cassert>

using namespace Torch;

///////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////
int main()
{
	//bool verbose = true;
	bool verbose = false;

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
	spFFT fft1d;
	print("Computing the FFT of x ...\n");
	fft1d.process(data);
	assert(fft1d.getNOutputs() == 1);
	fft1d.getOutput(0).print("F[x]");

	//
	spFFT ifft1d(true);
	print("Computing the iFFT of F[x]...\n");
	ifft1d.process(fft1d.getOutput(0));
	assert(ifft1d.getNOutputs() == 1);
	ifft1d.getOutput(0).print("inverse F[x]");

	//
	const int n_tests = 11;
	const int n_tests_N[n_tests] = { 2, 4, 8, 16, 32, 64, 128, 256, 1024, 2048, 4096};

	//for(int t = 0 ; t < 2 ; t++)
	for(int t = 0 ; t < n_tests ; t++)
	{
		print("Testing %d-points 1D FFT\n", n_tests_N[t]);

		N = n_tests_N[t];
		FloatTensor x(N);

		for(int i = 0 ; i < N ; i++)
		{
			double random_ = THRandom_uniform(0, 255);
	        	x(i) = random_;
		}

		if(verbose) x.print("x");

		//
		spFFT fft1d;
		print("  Computing the FFT of x ...\n");
		fft1d.process(x);
		if(verbose) fft1d.getOutput(0).print("F[x]");

		//
		spFFT ifft1d(true);
		print("  Computing the iFFT of F[x]...\n");
		ifft1d.process(fft1d.getOutput(0));
		if(verbose) ifft1d.getOutput(0).print("inverse F[x]");

		const FloatTensor& out = (const FloatTensor&) ifft1d.getOutput(0);

		double rmse = 0.0;
		for(int i = 0 ; i < N ; i++)
		{
		   	double z = x(i) - out.get(i);
			rmse += z*z;
		}
		rmse /= (double) N;

		print("RMSE = %g\n\n", rmse);
	}

	//
	print("\nOK\n");

	return 0;
}

