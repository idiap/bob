#include "ProbabilityDistribution.h"
#include "Tensor.h"
#include "File.h"

namespace Torch {

//////////////////////////////////////////////////////////////////////////
// Constructor

ProbabilityDistribution::ProbabilityDistribution()
{
	n_inputs = 0;
	n_parameters = 0;
	parameters = NULL;

	m_parameters->addI("n_inputs", 0, "number of inputs of the probability distribution");
	m_parameters->addI("n_parameters", 0, "number of parameters of the probability distribution");
	m_parameters->addDarray("parameters", 0, 0, "parameters of the probability distribution");
}

ProbabilityDistribution::ProbabilityDistribution(const int n_inputs_, const int n_parameters_)
{
	n_inputs = 0;
	n_parameters = 0;
	parameters = NULL;

	m_parameters->addI("n_inputs", 0, "number of inputs of the probability distribution");
	if(n_parameters_ > 0)
	{
		m_parameters->addI("n_parameters", n_parameters_, "number of parameters of the probability distribution");
		m_parameters->addDarray("parameters", n_parameters_, 0, "parameters of the probability distribution");
	}
	else
	{
		m_parameters->addI("n_parameters", 0, "number of parameters of the probability distribution");
		m_parameters->addDarray("parameters", 0, 0, "parameters of the probability distribution");
	}

	resize(n_inputs_);
}

//////////////////////////////////////////////////////////////////////////
// Destructor

ProbabilityDistribution::~ProbabilityDistribution()
{
}

bool ProbabilityDistribution::prepare()
{ 
	n_inputs = m_parameters->getI("n_inputs");
	n_parameters = m_parameters->getI("n_parameters");
	parameters = m_parameters->getDarray("parameters");

	return true; 
}
   
bool ProbabilityDistribution::resize(int n_inputs_, int n_parameters_)
{
	m_output.resize(1);

	//
	m_parameters->setI("n_inputs", n_inputs_);
	n_inputs = n_inputs_;
	if(n_parameters_ > 0)
	{
		m_parameters->setI("n_parameters", n_parameters_);
		n_parameters = n_parameters_;
		m_parameters->setDarray("parameters", n_parameters_);
		parameters = m_parameters->getDarray("parameters");
	}

	return true;
}

bool ProbabilityDistribution::forward(const Tensor& input)
{
	// Accept only 1D tensors of Double
	if (	input.nDimension() != 1 || input.getDatatype() != Tensor::Double)
	{
		warning("ProbabilityDistribution::forward() : incorrect number of dimensions or type.");
		
		return false;
	}
	if (	input.size(0) != n_inputs)
	{
		warning("ProbabilityDistribution::forward() : incorrect input size along dimension 0 (%d != %d).", input.size(0), n_inputs);
		
		return false;
	}

	DoubleTensor *t_input = (DoubleTensor *) &input;

	return forward(t_input);
}

///////////////////////////////////////////////////////////
// Loading/Saving the content from files (\emph{not the options}) - overriden

bool ProbabilityDistribution::loadFile(File& file)
{
	// Check the ID
	int id;
	if (file.taggedRead(&id, sizeof(int), 1, "ID") != 1)
	{
		Torch::message("ProbabilityDistribution::load - failed to read <ID> field!\n");
		return false;
	}
	if (id != getID())
	{
		Torch::message("ProbabilityDistribution::load - invalid <ID>, this is not the appropriate ProbabilityDistribution model!\n");
		return false;
	}

	if(m_parameters->loadFile(file) == false)
	{
	        Torch::message("ProbabilityDistribution::load - failed to load parameters\n");
		return false;
	}

	int n_inputs_ = m_parameters->getI("n_inputs");

	resize(n_inputs_);

	// OK
	return true;
}

bool ProbabilityDistribution::saveFile(File& file) const
{
	// Write the machine ID
	const int id = getID();
	if (file.taggedWrite(&id, sizeof(int), 1, "ID") != 1)
	{
		Torch::message("ProbabilityDistribution::save - failed to write <ID> field!\n");
		return false;
	}

	if(m_parameters->saveFile(file) == false)
	{
	        Torch::message("ProbabilityDistribution::load - failed to write parameters\n");
		return false;
	}

	// OK
	return true;
}

//////////////////////////////////////////////////////////////////////////
}
