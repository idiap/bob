#include "CmdLine.h"

#include "Machines.h"
#include "MSECriterion.h"

using namespace Torch;


int main(int argc, char* argv[])
{
	int n_t = 4;
	int n_input = 5;
	int n_hu = 3;
	int n_output = 1;

	bool verbose = true;

	THRandom_manualSeed(950305);

	GradientMachine *gm_linear_1 = new Linear(n_input, n_hu);
	GradientMachine *gm_tanh_1 = new Tanh(n_hu);
	GradientMachine *gm_linear_2 = new Linear(n_hu, n_output);
	GradientMachine *gm_tanh_2 = new Tanh(n_output);

	gm_linear_1->prepare();
	gm_linear_1->shuffle();
	gm_linear_2->prepare();
	gm_linear_2->shuffle();

	DoubleTensor *T = new DoubleTensor(n_input);

	GradientMachine *gm;
	const DoubleTensor *input;
	
	for(int t = 0 ; t < n_t ; t++)
	{
		for(int j = 0 ; j < n_input ; j++)
		{
			double random_ = THRandom_uniform(0, 1);
			T->set(j, random_);
		}

		if(verbose) T->print("Input Tensor");

		input = T;
		gm = gm_linear_1;
		gm->forward(*input);
		if(verbose) gm->getOutput().sprint("Linear 1");

		input = &gm->getOutput();
		gm = gm_tanh_1;
		gm->forward(*input);
		if(verbose) gm->getOutput().sprint("Tanh 1");

		input = &gm->getOutput();
		gm = gm_linear_2;
		gm->forward(*input);
		if(verbose) gm->getOutput().sprint("Linear 2");

		input = &gm->getOutput();
		gm = gm_tanh_2;
		gm->forward(*input);
		if(verbose) gm->getOutput().sprint("Tanh 2");

	}


	delete T;

	delete gm_tanh_2;
	delete gm_linear_2;
	delete gm_tanh_1;
	delete gm_linear_1;

	// OK
	print("OK.\n");

	return 0;
}

