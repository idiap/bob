#include "torch5spro.h"

using namespace Torch;


int main(int argc, char* argv[])
{
	int n_gm = 4;
	int n_t = 10;
	int n_frames = 50;
	bool verbose = true;

	GradientMachine **gm = new GradientMachine* [n_gm];

	THRandom_manualSeed(950305);

	gm[0] = new Exp();
	gm[1] = new Log();
	gm[2] = new Sigmoid();
	gm[3] = new Tanh();

	GradientMachine *linear = new Linear();
	linear->prepare();
	linear->shuffle();

	for(int t = 0 ; t < n_t ; t++)
	{
	  int m = (int) THRandom_uniform(1, n_frames);

	  print("Test %d with %d frames ...\n", t, n_frames);

	  gm[0]->resize(m, m, 0);
	  gm[0]->prepare();

          DoubleTensor *T = new DoubleTensor(m);

	  for(int j = 0 ; j < m ; j++)
	  {
	  	double random_ = THRandom_uniform(0, 1);
	  	T->set(j, random_);
	  }

	  if(verbose)
	  	T->print("Input Tensor");

	  gm[0]->forward(*T);

	  if(verbose)
	  	gm[0]->getOutput().sprint("Output %d", 0);

	  for(int i = 1 ; i < n_gm ; i++)
	  {
	  	gm[i]->resize(m, m, 0);
	  	gm[i]->prepare();

		gm[i]->forward(gm[i-1]->getOutput());

	  	if(verbose)
			gm[i]->getOutput().sprint("Output %d", i);
	  }

	  int n = 2;
	  linear->resize(m, n, (m+1)*n);
	  linear->prepare();
	  linear->shuffle();
	  linear->forward(gm[n_gm-1]->getOutput());
	  if(verbose)
	  	linear->getOutput().print("Output linear");

	  delete T;
	}

	for(int i = 0 ; i < n_gm ; i++) delete gm[i];
	delete [] gm;

	// OK
	print("OK.\n");

	return 0;
}

