#include "torch5spro.h"

using namespace Torch;

int main(int argc, char* argv[])
{
	int n_t = 1;
	int n_inputs = 4;
	int n_gaussians = 2;

	bool verbose = true;

	unsigned long seed = THRandom_seed();
	print("Seed = %ld\n", seed);
	//THRandom_manualSeed(950305);

	ProbabilityDistribution *D = new MultiVariateDiagonalGaussianDistribution(n_inputs, n_gaussians);

	D->setBOption("log mode", true);
	D->shuffle();
	D->prepare();
	
	if(verbose) D->print();

	DoubleTensor *X = new DoubleTensor(n_inputs);
	double L = 0.0;

	for(int t = 0 ; t < n_t ; t++)
	{
		for(int j = 0 ; j < n_inputs ; j++)
		{
			double random_ = THRandom_uniform(0, 1);
			X->set(j, random_);
		}

		if(verbose) X->print("Input Tensor");

		D->forward(*X);
		if(verbose) D->getOutput().sprint("P(X | D)");

		L += D->getOutput().get(0);
	}
	L /= (double) n_t;

	print("L = %g (%d x %d-%d)\n", L, n_t, n_inputs, n_gaussians);

	delete X;

	delete D;

	// OK
	print("OK.\n");

	return 0;
}

