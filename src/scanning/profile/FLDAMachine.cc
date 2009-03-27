#include "FLDAMachine.h"
#include "File.h"

namespace Torch
{

/////////////////////////////////////////////////////////////////////////
// Constructor

FLDAMachine::FLDAMachine(int size)
	:	Machine(),
		m_size(0),
		m_proj(0), m_proj_avg(0.0),
		m_threshold(0.0)
{
	m_output = new DoubleTensor(1);
	m_poutput = (double*)m_output->dataW();

	resize(size);
}

/////////////////////////////////////////////////////////////////////////
// Destructor

FLDAMachine::~FLDAMachine()
{
	delete m_output;
	delete[] m_proj;
}

/////////////////////////////////////////////////////////////////////////
// Resize

bool FLDAMachine::resize(int size)
{
	if (size < 1)
	{
		return false;
	}

	if (size != m_size)
	{
		m_size = size;
		delete[] m_proj;
		m_proj = new double[m_size];
		for (int i = 0; i < m_size; i ++)
		{
			m_proj[i] = 0.0;
		}

		m_proj_avg = 0.0;
	}

	return true;
}

/////////////////////////////////////////////////////////////////////////
// Set machine's parameters

void FLDAMachine::setThreshold(double threshold)
{
	m_threshold = threshold;
}

void FLDAMachine::setProjection(const double* proj, double proj_avg)
{
	for (int i = 0; i < m_size; i ++)
	{
		m_proj[i] = proj[i];
	}
	m_proj_avg = proj_avg;
}

/////////////////////////////////////////////////////////////////////////
// Process the input tensor

bool FLDAMachine::forward(const Tensor& input)
{
	// Check input type
	if (	input.getDatatype() != Tensor::Double ||
		input.nDimension() != 1 ||
		input.size(0) != m_size)
	{
		message("FLDAMachine::forward - invalid input tensor!\n");
		return false;
	}

	// OK
	const double* data = (const double*)input.dataR();

	// Project the input vector
	double sum = 0.0;
	for (int i = 0; i < m_size; i ++)
	{
		sum += m_proj[i] * data[i];
	}

	// Substract the projected average
	*m_poutput = sum - m_proj_avg;
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Loading/Saving the content from files (\emph{not the options})

bool FLDAMachine::loadFile(File& file)
{
	// Check the ID
	int id;
	if (file.taggedRead(&id, sizeof(int), 1, "ID") != 1)
	{
		Torch::message("FLDAMachine::load - failed to read <ID> field!\n");
		return false;
	}
	if (id != getID())
	{
		Torch::message("FLDAMachine::load - invalid <ID>, this is not a FLDAMachine model!\n");
		return false;
	}

	// Read the size
	int size;
	if (file.taggedRead(&size, sizeof(int), 1, "SIZE") != 1)
	{
		Torch::message("FLDAMachine::load - failed to read <SIZE> field!\n");
		return false;
	}
	if (resize(size) == false)
	{
		Torch::message("FLDAMachine::load - invalid <SIZE> field!\n");
		return false;
	}

	// Read the projection vector and the projected average
	if (file.taggedRead(m_proj, sizeof(double), m_size, "PROJ") != m_size)
	{
		Torch::message("FLDAMachine::load - failed to read <PROJ> field!\n");
		return false;
	}

	if (file.taggedRead(&m_proj_avg, sizeof(double), 1, "PROJ_AVG") != 1)
	{
		Torch::message("FLDAMachine::load - failed to read <PROJ_AVG> field!\n");
		return false;
	}

	// Read the threshold
	if (file.taggedRead(&m_threshold, sizeof(double), 1, "THRESHOLD") != 1)
	{
		Torch::message("FLDAMachine::load - failed to read <THRESHOLD> field!\n");
		return false;
	}

	// OK
	return true;
}

bool FLDAMachine::saveFile(File& file) const
{
	// Write the ID
	const int id = getID();
	if (file.taggedWrite(&id, sizeof(int), 1, "ID") != 1)
	{
		Torch::message("FLDAMachine::save - failed to write <ID> field!\n");
		return false;
	}

	// Write the size
	if (file.taggedWrite(&m_size, sizeof(int), 1, "SIZE") != 1)
	{
		Torch::message("FLDAMachine::save - failed to write <SIZE> field!\n");
		return false;
	}

	// Write the projection vector and the projected average
	if (file.taggedWrite(m_proj, sizeof(double), m_size, "PROJ") != m_size)
	{
		Torch::message("FLDAMachine::save - failed to write <PROJ> field!\n");
		return false;
	}
	if (file.taggedWrite(&m_proj_avg, sizeof(double), 1, "PROJ_AVG") != 1)
	{
		Torch::message("FLDAMachine::save - failed to write <PROJ_AVG> field!\n");
		return false;
	}

	// Write the threshold
	if (file.taggedWrite(&m_threshold, sizeof(double), 1, "THRESHOLD") != 1)
	{
		Torch::message("FLDAMachine::save - failed to write <THRESHOLD> field!\n");
		return false;
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////

}
