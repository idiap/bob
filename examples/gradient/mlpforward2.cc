#include "CmdLine.h"

#include "Machines.h"

using namespace Torch;

int main(int argc, char* argv[])
{
	int n_t = 4;
	int n_input = 5;
	int n_hu = 3;
	int n_output = 1;

	bool verbose = true;

	THRandom_manualSeed(950305);

	GradientMachine *mlp = new MLP(n_input, n_hu, n_output);

	mlp->prepare();
	mlp->shuffle();

	DoubleTensor *T = new DoubleTensor(n_input);

	for(int t = 0 ; t < n_t ; t++)
	{
		for(int j = 0 ; j < n_input ; j++)
		{
			double random_ = THRandom_uniform(0, 1);
			T->set(j, random_);
		}

		if(verbose) T->print("Input Tensor");

		mlp->forward(*T);
		if(verbose) mlp->getOutput().sprint("MLP");

	}


	delete T;

	delete mlp;

	// OK
	print("OK.\n");

	return 0;
}

