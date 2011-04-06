#include "machine/GradientMachine.h"
#include "core/Tensor.h"
#include "core/File.h"

namespace Torch {

//////////////////////////////////////////////////////////////////////////
// Constructor

GradientMachine::GradientMachine()
{
	m_beta = NULL;

	n_inputs = 0;
	n_outputs = 0;
	n_parameters = 0;
	parameters = NULL;
	der_parameters = NULL;

	m_parameters->addI("n_inputs", 0, "number of inputs of the gradient machine");
	m_parameters->addI("n_outputs", 0, "number of outputs of the gradient machine");
	m_parameters->addI("n_parameters", 0, "number of parameters of the gradient machine");
	m_parameters->addDarray("parameters", 0, 0, "parameters of the gradient machine");
	m_parameters->addDarray("der_parameters", 0, 0, "derivatives of the parameters of the gradient machine");
}

GradientMachine::GradientMachine(const int n_inputs_, const int n_outputs_, const int n_parameters_)
{
	m_beta = NULL;

	n_inputs = 0;
	n_outputs = 0;
	n_parameters = 0;
	parameters = NULL;
	der_parameters = NULL;

	m_parameters->addI("n_inputs", 0, "number of inputs of the gradient machine");
	m_parameters->addI("n_outputs", 0, "number of outputs of the gradient machine");
	if(n_parameters_ > 0)
	{
		m_parameters->addI("n_parameters", n_parameters_, "number of parameters of the gradient machine");
		m_parameters->addDarray("parameters", n_parameters_, 0, "parameters of the gradient machine");
		m_parameters->addDarray("der_parameters", n_parameters_, 0, "derivatives of the parameters of the gradient machine");
	}
	else
	{
		m_parameters->addI("n_parameters", 0, "number of parameters of the gradient machine");
		m_parameters->addDarray("parameters", 0, 0, "parameters of the gradient machine");
		m_parameters->addDarray("der_parameters", 0, 0, "derivatives of the parameters of the gradient machine");
	}

	resize(n_inputs_, n_outputs_);
}

//////////////////////////////////////////////////////////////////////////
// Destructor

GradientMachine::~GradientMachine()
{
        // Cleanup
	delete m_beta;
}

bool GradientMachine::prepare()
{
	n_inputs = m_parameters->getI("n_inputs");
	n_outputs = m_parameters->getI("n_outputs");
	n_parameters = m_parameters->getI("n_parameters");
	parameters = m_parameters->getDarray("parameters");
	der_parameters = m_parameters->getDarray("der_parameters");

	return true;
}

bool GradientMachine::resize(int n_inputs_, int n_outputs_, int n_parameters_)
{
	m_output.resize(n_outputs_);

	// Free
	if(m_beta != NULL)
	{
		delete m_beta;
		m_beta = NULL;
	}
	// Reallocate
	m_beta = new DoubleTensor(n_inputs_);

	//
	m_parameters->setI("n_inputs", n_inputs_);
	n_inputs = n_inputs_;
	m_parameters->setI("n_outputs", n_outputs_);
	n_outputs = n_outputs_;
	if(n_parameters_ > 0)
	{
		m_parameters->setI("n_parameters", n_parameters_);
		n_parameters = n_parameters_;
		m_parameters->setDarray("parameters", n_parameters_);
		parameters = m_parameters->getDarray("parameters");
		m_parameters->setDarray("der_parameters", n_parameters_);
		der_parameters = m_parameters->getDarray("der_parameters");
	}

	return true;
}

bool GradientMachine::forward(const Tensor& input)
{
	//int n_inputs = m_parameters->getI("n_inputs");

	// Accept only 1D tensors of Double
	if (	input.nDimension() != 1 || input.getDatatype() != Tensor::Double)
	{
		warning("GradientMachine::forward() : incorrect number of dimensions or type.");

		return false;
	}
	if (	input.size(0) != n_inputs)
	{
		warning("GradientMachine::forward() : incorrect input size along dimension 0 (%d != %d).", input.size(0), n_inputs);

		return false;
	}

	const DoubleTensor *t_input = (const DoubleTensor *) &input;

	return forward(t_input);
}

bool GradientMachine::backward(const Tensor& input, const DoubleTensor *alpha)
{
	//int n_inputs = m_parameters->getI("n_inputs");

	// Accept only 1D tensors of Double
	if (	input.nDimension() != 1 || input.getDatatype() != Tensor::Double)
	{
		warning("GradientMachine::backward() : incorrect number of dimensions or type.");

		return false;
	}
	if (	input.size(0) != n_inputs)
	{
		warning("GradientMachine::backward() : incorrect input size along dimension 0 (%d != %d).", input.size(0), n_inputs);

		return false;
	}

	DoubleTensor *t_input = (DoubleTensor *) &input;

	return backward(t_input, alpha);
}

///////////////////////////////////////////////////////////
// Loading/Saving the content from files (\emph{not the options}) - overriden

bool GradientMachine::loadFile(File& file)
{
	// Check the ID
	int id;
	if (file.taggedRead(&id, 1, "ID") != 1)
	{
		Torch::message("GradientMachine::load - failed to read <ID> field!\n");
		return false;
	}
	if (id != getID())
	{
		Torch::message("GradientMachine::load - invalid <ID>, this is not the appropriate GradientMachine model!\n");
		return false;
	}

	if(m_parameters->loadFile(file) == false)
	{
	        Torch::message("GradientMachine::load - failed to load parameters\n");
		return false;
	}

	int n_inputs_ = m_parameters->getI("n_inputs");
	int n_outputs_ = m_parameters->getI("n_outputs");

	resize(n_inputs_, n_outputs_);

	//m_parameters->print("Gradient Machine");

	// OK
	return true;
}

bool GradientMachine::saveFile(File& file) const
{
	// Write the machine ID
	const int id = getID();
	if (file.taggedWrite(&id, 1, "ID") != 1)
	{
		Torch::message("GradientMachine::save - failed to write <ID> field!\n");
		return false;
	}

	if(m_parameters->saveFile(file) == false)
	{
	        Torch::message("GradientMachine::load - failed to write parameters\n");
		return false;
	}

	//m_parameters->print("Gradient Machine");

	// OK
	return true;
}

//////////////////////////////////////////////////////////////////////////
}
