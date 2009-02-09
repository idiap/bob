#include "spDCT.h"
#include "Tensor.h"
#include <cassert>

using namespace Torch;

///////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////
int main()
{

   	THRandom_manualSeed(950305);

	print("\n\nTesting 1D DCT\n\n");

	//
	int N = 4;

	FloatTensor data(N);

	for(int i = 0 ; i < N ; i++)
	{
		double random_ = THRandom_uniform(0, 255);
	        data(i) = random_;
	}

	data.print("x");
	
	//
	spDCT dct1d;
	print("Computing the DCT of x ...\n");
	dct1d.process(data);
	assert(dct1d.getNOutputs() == 1);
	dct1d.getOutput(0).print("F[x]");

	//
	spDCT idct1d(true);
	print("Computing the iDCT of F[x]...\n");
	idct1d.process(dct1d.getOutput(0));
	assert(idct1d.getNOutputs() == 1);
	idct1d.getOutput(0).print("inverse F[x]");


	//
	const int n_tests = 11;
	const int n_tests_N[n_tests] = { 2, 4, 8, 16, 32, 64, 128, 256, 1024, 2048, 4096};

	for(int t = 0 ; t < n_tests ; t++)
	{
		print("Testing %d-points 1D DCT\n", n_tests_N[t]);

		N = n_tests_N[t];

		FloatTensor x(N);

		for(int i = 0 ; i < N ; i++)
		{
			double random_ = THRandom_uniform(0, 255);
		        x(i) = random_;
		}

		//
		spDCT dct1d;
		print("Computing the DCT of x ...\n");
		dct1d.process(x);

		//
		spDCT idct1d(true);
		print("Computing the iDCT of F[x]...\n");
		idct1d.process(dct1d.getOutput(0));

		const FloatTensor& out = (const FloatTensor&) idct1d.getOutput(0);

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

