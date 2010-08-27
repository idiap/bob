#include "machine/Linear.h"

namespace Torch {

//////////////////////////////////////////////////////////////////////////
// Constructor

Linear::Linear() : GradientMachine()
{
	addFOption("weight decay", 0.0, "Weight decay");
	addFOption("momentum", 0.0, "Inertia momentum");

	weights = NULL;
	bias = NULL;
	der_weights = NULL;
	der_bias = NULL;
	delta_parameters = NULL;
}

Linear::Linear(const int n_inputs_, const int n_outputs_)
   	: GradientMachine(n_inputs_, n_outputs_, (n_inputs_+1)*n_outputs_)
{
	addFOption("weight decay", 0.0, "Weight decay");
	addFOption("momentum", 0.0, "Inertia momentum");

	double *parameters_ = m_parameters->getDarray("parameters");
	double *der_parameters_ = m_parameters->getDarray("der_parameters");

	weights = parameters_;
	bias = parameters_ + n_inputs_*n_outputs_;
	der_weights = der_parameters_;
	der_bias = der_parameters_ + n_inputs_*n_outputs_;

	delta_parameters = new double [(n_inputs_+1)*n_outputs_];
}

//////////////////////////////////////////////////////////////////////////
// Destructor

Linear::~Linear()
{
	if(delta_parameters != NULL) delete [] delta_parameters;
}


//////////////////////////////////////////////////////////////////////////

bool Linear::resize(int n_inputs_, int n_outputs_, int n_parameters_)
{
   	GradientMachine::resize(n_inputs_, n_outputs_, n_parameters_);

	double *parameters_ = m_parameters->getDarray("parameters");
	double *der_parameters_ = m_parameters->getDarray("der_parameters");

	weights = parameters_;
	bias = parameters_ + n_inputs_*n_outputs_;
	der_weights = der_parameters_;
	der_bias = der_parameters_ + n_inputs_*n_outputs_;

	if(delta_parameters != NULL) delete [] delta_parameters;
	delta_parameters = new double [(n_inputs_+1)*n_outputs_];

	return true;
}

bool Linear::shuffle()
{
	//int n_inputs = m_parameters->getI("n_inputs");
	//int n_outputs = m_parameters->getI("n_outputs");
	double *parameters_ = m_parameters->getDarray("parameters");
	double *der_parameters_ = m_parameters->getDarray("der_parameters");

	weights = parameters_;
	bias = parameters_ + n_inputs*n_outputs;
	der_weights = der_parameters_;
	der_bias = der_parameters_ + n_inputs*n_outputs;

	double *weights_ = weights;
	double *der_weights_ = der_weights;
	double bound = 1./sqrt((double)n_inputs);
	for(int i = 0; i < n_outputs; i++)
	{
		for(int j = 0; j < n_inputs; j++)
		{
			weights_[j] = THRandom_uniform(-bound, bound);
			der_weights_[j] = 0.0;
		}
		weights_ += n_inputs;
		der_weights_ += n_inputs;
		bias[i] = THRandom_uniform(-bound, bound);
		der_bias[i] = 0.0;
	}

	return true;
}

bool Linear::prepare()
{
	int n_parameters_ = m_parameters->getI("n_parameters");
   
	if(delta_parameters != NULL)
		for(int i = 0; i < n_parameters_; i++) delta_parameters[i] = 0.0;

	return GradientMachine::prepare();
}

bool Linear::Ginit()
{
	//int n_parameters = m_parameters->getI("n_parameters");
	//double *der_parameters = m_parameters->getDarray("der_parameters");

	for(int i = 0; i < n_parameters; i++) der_parameters[i] = 0.0;

	return true;
}

bool Linear::Gupdate(double learning_rate)
{
	//int n_parameters = m_parameters->getI("n_parameters");
	//double *parameters = m_parameters->getDarray("parameters");
	//double *der_parameters = m_parameters->getDarray("der_parameters");

	float momentum = getFOption("momentum");

	for(int i = 0; i < n_parameters; i++) 
	{
		double z = parameters[i];
		parameters[i] -= learning_rate * der_parameters[i];

		if(momentum != 0)
		{
			delta_parameters[i] = (parameters[i] - z);
			parameters[i] += momentum * delta_parameters[i];
		}
	}
	
	return true;
}
		
bool Linear::forward(const DoubleTensor *input)
{
	double *weights_ = weights;
	for(int i = 0; i < n_outputs; i++)
	{
		double z = bias[i];
		for(int j = 0; j < n_inputs; j++)
			z += weights_[j] * (*input)(j);
		weights_ += n_inputs;
		m_output(i) = z;
	}
	return true;
}

bool Linear::backward(const DoubleTensor *input, const DoubleTensor *alpha)
{
	//int n_inputs = m_parameters->getI("n_inputs");
	//int n_outputs = m_parameters->getI("n_outputs");

	for(int i = 0; i < n_inputs; i++) (*m_beta)(i) = 0.0;

	double *weights_ = weights;
	double *der_weights_ = der_weights;
	for(int i = 0; i < n_outputs; i++)
	{
		double z = (*alpha)(i);
		for(int j = 0; j < n_inputs; j++)
		{
			(*m_beta)(j) += z * weights_[j];
			der_weights_[j] += z * (*input)(j);
		}
		weights_ += n_inputs;
		der_weights_ += n_inputs;
		der_bias[i] += z;
	}
	
	float weight_decay = getFOption("weight decay");
	if(weight_decay != 0)
	{
		//double *src_ = m_parameters->getDarray("parameters");
		double *src_ = parameters;
		//double *dst_ = m_parameters->getDarray("der_parameters");
		double *dst_ = der_parameters;
		for(int i = 0; i < n_inputs*n_outputs;i++)
			dst_[i] += weight_decay * src_[i];
	}
  m_beta->resetFromData();
	return true;
}

//////////////////////////////////////////////////////////////////////////
}
