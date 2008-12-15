#include "GradientMachine.h"
#include "Tensor.h"
#include "File.h"

namespace Torch {

//////////////////////////////////////////////////////////////////////////
// Constructor

GradientMachine::GradientMachine(int n_inputs_, int n_outputs_, int n_parameters_)
   	: Machine(), n_inputs(n_inputs_), n_outputs(n_outputs_), n_parameters(n_parameters_)
{
	// Allocate the output
	m_output = new DoubleTensor(n_outputs);
	m_parameters = NULL;
	m_der_parameters = NULL;
	if(n_parameters > 0)
	{
		m_parameters = new double [n_parameters]; 
		m_der_parameters = new double [n_der_parameters]; 
	}
}

//////////////////////////////////////////////////////////////////////////
// Destructor

GradientMachine::~GradientMachine()
{
        // Cleanup
	if(m_parameters != NULL) delete[] m_parameters;
	if(m_der_parameters != NULL) delete[] m_der_parameters;
	delete m_output;
}


bool GradientMachine::forward(const Tensor& input)
{
   	/*
		unroll the input tensor:
			if 1D then process as one frame
			if 2D then process as a sequence of frames
			if 3D ???
	*/

	//m_output->set(0, m_lut[lbp]);
	//m_output->zero();
	//m_output->one();
	//m_output->fill();
	return true;
}

///////////////////////////////////////////////////////////
// Loading/Saving the content from files (\emph{not the options}) - overriden

bool GradientMachine::loadFile(File& file)
{
	// Check the ID
	int id;
	if (file.taggedRead(&id, sizeof(int), 1, "ID") != 1)
	{
		Torch::message("GradientMachine::load - failed to read <ID> field!\n");
		return false;
	}
	if (id != getID())
	{
		Torch::message("GradientMachine::load - invalid <ID>, this is not the appropriate GradientMachine model!\n");
		return false;
	}

	// Read the machine parameters
	const int ret = file.taggedRead(m_parameters, sizeof(double), m_parameters, "PARAMETERS");
	if (ret != m_parameters)
	{
	        Torch::message("GradientMachine::load - failed to read <PARAMETERS> field!\n");
		return false;
	}

	// OK
	return true;
}

bool GradientMachine::saveFile(File& file) const
{
	// Write the machine ID
	const int id = getID();
	if (file.taggedWrite(&id, sizeof(int), 1, "ID") != 1)
	{
		Torch::message("GradientMachine::save - failed to write <ID> field!\n");
		return false;
	}

	// Write the machine parameters
	if (file.taggedWrite(m_parameters, sizeof(double), m_parameters, "PARAMETERS") != m_parameters)
	{
		Torch::message("GradientMachine::save - failed to write <PARAMETERS> field!\n");
		return false;
	}

	// OK
	return true;
}

//////////////////////////////////////////////////////////////////////////
}
