#include "torch5spro.h"

using namespace Torch;


int main(int argc, char* argv[])
{
	int n_iter = 5000;
	int n_input = 2;
	int n_hu = 2;
	int n_output = 1;
	float learning_rate = 0.01;
	float learning_rate_decay = 0.0;
	float weight_decay = 0.0;
	double pT = 0.9;
	double nT = -0.9;
	//double pT = 0.2;
	//double nT = 0.8;

	bool verbose = false;

	//unsigned long seed = THRandom_seed();
	//print("Seed = %ld\n", seed);
	//THRandom_manualSeed(950305);
	THRandom_manualSeed(503059);

	GradientMachine *gm_linear_1 = new Linear(n_input, n_hu);
	GradientMachine *gm_nonlinear_1 = new Tanh(n_hu);
	//GradientMachine *gm_nonlinear_1 = new Sigmoid(n_hu);
	GradientMachine *gm_linear_2 = new Linear(n_hu, n_output);
	GradientMachine *gm_nonlinear_2 = new Tanh(n_output);
	//GradientMachine *gm_nonlinear_2 = new Sigmoid(n_output);

	gm_linear_1->prepare();
	gm_linear_1->shuffle();
	gm_linear_1->m_parameters->print("linear 1");
	gm_linear_1->setFOption("weight decay", weight_decay);
	gm_linear_2->prepare();
	gm_linear_2->shuffle();
	gm_linear_2->m_parameters->print("linear 2");
	gm_linear_2->setFOption("weight decay", weight_decay);

	DoubleTensor **data = new DoubleTensor* [4];
	DoubleTensor **target = new DoubleTensor* [4];

	//
	data[0] = new DoubleTensor(n_input);
	target[0] = new DoubleTensor(1);
	data[0]->set(0, 0);
	data[0]->set(1, 0);
	target[0]->set(0, nT);
	//
	data[1] = new DoubleTensor(n_input);
	target[1] = new DoubleTensor(1);
	data[1]->set(0, 0);
	data[1]->set(1, 1);
	target[1]->set(0, pT);
	//
	data[2] = new DoubleTensor(n_input);
	target[2] = new DoubleTensor(1);
	data[2]->set(0, 1);
	data[2]->set(1, 0);
	target[2]->set(0, pT);
	//
	data[3] = new DoubleTensor(n_input);
	target[3] = new DoubleTensor(1);
	data[3]->set(0, 1);
	data[3]->set(1, 1);
	target[3]->set(0, nT);

	//
	Criterion *criterion = new MSECriterion(1);
	//Criterion *criterion = new TwoClassNLLCriterion();

	//
	GradientMachine *gm;
	const DoubleTensor *input;

	double current_learning_rate = learning_rate;

	for(int iter = 0 ; iter < n_iter ; iter++)
	{
		double mse = 0.0;

		for(int t = 0 ; t < 4 ; t++)
		{
			// Init derivative parameters
			gm_linear_1->Ginit();
			gm_linear_2->Ginit();

			input = data[t];

			if(verbose)
				input->sprint("data[%d]", t);

			// Forward
			gm = gm_linear_1;
			gm->forward(*input);
			if(verbose) gm->getOutput().sprint("Linear 1");

			input = &gm->getOutput();
			gm = gm_nonlinear_1;
			gm->forward(*input);
			if(verbose) gm->getOutput().sprint("NonLinear 1");

			input = &gm->getOutput();
			gm = gm_linear_2;
			gm->forward(*input);
			if(verbose) gm->getOutput().sprint("Linear 2");

			input = &gm->getOutput();
			gm = gm_nonlinear_2;
			gm->forward(*input);
			if(verbose) gm->getOutput().sprint("NonLinear 2");

			if(verbose)
				gm->getOutput().sprint("output");

			criterion->forward(&gm->getOutput(), target[t]);

			if(verbose)
			{
				criterion->m_target->print("target");
				criterion->m_error->print("MSE");
				criterion->m_beta->print("beta");
			}

			mse += criterion->m_error->get(0);

			// Backward
			input = data[t];

			gm_nonlinear_2->backward(NULL, criterion->m_beta);
			gm_linear_2->backward(&gm_nonlinear_1->getOutput(), gm_nonlinear_2->m_beta);
			gm_nonlinear_1->backward(NULL, gm_linear_2->m_beta);
			gm_linear_1->backward(input, gm_nonlinear_1->m_beta);


			// update parameters
			gm_linear_1->Gupdate(current_learning_rate);
			gm_linear_2->Gupdate(current_learning_rate);
		}

		print("MSE = %d %g (%g)\n", iter, mse, current_learning_rate);
		current_learning_rate = learning_rate/(1.+((float)(iter))*learning_rate_decay);
	}

	gm_linear_1->m_parameters->print("linear 1");
	gm_linear_2->m_parameters->print("linear 2");

	for(int t = 0 ; t < 4 ; t++)
	{
		input = data[t];

		input->sprint("data[%d]", t);

		// Forward
		gm = gm_linear_1;
		gm->forward(*input);

		input = &gm->getOutput();
		gm = gm_nonlinear_1;
		gm->forward(*input);

		input = &gm->getOutput();
		gm = gm_linear_2;
		gm->forward(*input);

		input = &gm->getOutput();
		gm = gm_nonlinear_2;
		gm->forward(*input);

		gm->getOutput().sprint("output");
		target[t]->print("target");
	}

	delete criterion;

	delete data[0];
	delete data[1];
	delete data[2];
	delete data[3];
	delete target[0];
	delete target[1];
	delete target[2];
	delete target[3];

	delete []data;
	delete []target;

	delete gm_nonlinear_2;
	delete gm_linear_2;
	delete gm_nonlinear_1;
	delete gm_linear_1;

	// OK
	print("OK.\n");

	return 0;
}

