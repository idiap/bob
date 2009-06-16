#include "MLP.h"

namespace Torch {

//////////////////////////////////////////////////////////////////////////
// Constructor

MLP::MLP() : GradientMachine()
{
	addFOption("weight decay", 0.0, "Weight decay");
	addFOption("momentum", 0.0, "Inertia momentum");

	m_parameters->addD("nhu", 0, "Number of hidden units");

	n_gm = 0;
	gm = NULL;
}

MLP::MLP(const int n_inputs_, const int n_hidden_units_, const int n_outputs_)
   	//: GradientMachine(n_inputs_, n_outputs_, (n_inputs_+1)*n_hidden_units_ + (n_hidden_units_+1)*n_outputs_)
   	: GradientMachine(n_inputs_, n_outputs_)
{
	addFOption("weight decay", 0.0, "Weight decay");
	addFOption("momentum", 0.0, "Inertia momentum");

	m_parameters->addD("nhu", n_hidden_units_, "Number of hidden units");

	n_gm = 4;
	gm = new GradientMachine* [4];
	gm[0] = new Linear(n_inputs_, n_hidden_units_);
	gm[1] = new Tanh(n_hidden_units_);
	gm[2] = new Linear(n_hidden_units_, n_outputs_);
	gm[3] = new Tanh(n_outputs_);
}

//////////////////////////////////////////////////////////////////////////
// Destructor

MLP::~MLP()
{
	if(gm != NULL)
	{
	   	delete gm[0];
	   	delete gm[1];
	   	delete gm[2];
	   	delete gm[3];
		delete []gm;
	}
}


//////////////////////////////////////////////////////////////////////////

bool MLP::prepare()
{
	// set the weight decay to the Linear machines
	float weight_decay = getFOption("weight decay");

	gm[0]->setFOption("weight decay", weight_decay);
	gm[2]->setFOption("weight decay", weight_decay);

	float momentum = getFOption("momentum");

	gm[0]->setFOption("momentum", momentum);
	gm[2]->setFOption("momentum", momentum);

   	for(int i = 0 ; i < n_gm ; i++) gm[i]->prepare();

	return true;
}

bool MLP::shuffle()
{
   	// randomize the Linear machines
	gm[0]->shuffle();
	gm[2]->shuffle();

	return true;
}

bool MLP::Ginit()
{
	// reset the Linear machine derivatives to zero
	gm[0]->Ginit();
	gm[2]->Ginit();

	return true;
}

bool MLP::Gupdate(double learning_rate)
{
	gm[0]->Gupdate(learning_rate);
	gm[2]->Gupdate(learning_rate);

	return true;
}

bool MLP::forward(const DoubleTensor *input)
{
	gm[0]->forward(*input);
	gm[1]->forward(gm[0]->getOutput());
	gm[2]->forward(gm[1]->getOutput());
	gm[3]->forward(gm[2]->getOutput());

	//int n_outputs = m_parameters->getI("n_outputs");

	double *src = (double *) gm[3]->getOutput().dataR();
	double *dst = (double *) m_output.dataW();

	for(int i = 0; i < n_outputs; i++) dst[i] = src[i];

	return true;
}

bool MLP::backward(const DoubleTensor *input, const DoubleTensor *alpha)
{
	//double *alpha_ = (double *) alpha->dataR();

	//
	// backward alpha (criterion's output) to the connected machines
	gm[3]->backward(NULL, alpha);
	gm[2]->backward(&gm[1]->getOutput(), gm[3]->m_beta);
	gm[1]->backward(NULL, gm[2]->m_beta);
	gm[0]->backward(input, gm[1]->m_beta);

	return true;
}

bool MLP::loadFile(File& file)
{
   	print("Loading MLP from file ...\n");
	
	// Check the ID
	int id;
	if (file.taggedRead(&id, sizeof(int), 1, "ID") != 1)
	{
		Torch::message("MLP::load - failed to read <ID> field!\n");
		return false;
	}
	if (id != getID())
	{
		Torch::message("MLP::load - invalid <ID>, this is not the appropriate GradientMachine model!\n");
		return false;
	}

	n_gm = 4;
	gm = new GradientMachine* [4];
	gm[0] = new Linear();
	gm[1] = new Tanh();
	gm[2] = new Linear();
	gm[3] = new Tanh();

   	for(int i = 0 ; i < n_gm ; i++) gm[i]->loadFile(file);

	resize(gm[0]->m_parameters->getI("n_inputs"), gm[3]->m_parameters->getI("n_outputs"));

	// OK
	return true;
}

bool MLP::saveFile(File& file) const
{
   	print("Saving MLP from file ...\n");
	
	// Write the machine ID
	const int id = getID();
	if (file.taggedWrite(&id, sizeof(int), 1, "ID") != 1)
	{
		Torch::message("MLP::save - failed to write <ID> field!\n");
		return false;
	}

   	for(int i = 0 ; i < n_gm ; i++) gm[i]->saveFile(file);

	// OK
	return true;
}

//////////////////////////////////////////////////////////////////////////
}
