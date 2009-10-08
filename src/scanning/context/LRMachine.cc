#include "LRMachine.h"
#include "File.h"

namespace Torch
{

/////////////////////////////////////////////////////////////////////////
// Constructor

LRMachine::LRMachine(int size)
	:	Machine(),
		m_size(0),
		m_weights(0),
		m_threshold(0.5)
{
	m_output.resize(1);
	m_poutput = (double*)m_output.dataW();

	resize(size);
}

/////////////////////////////////////////////////////////////////////////
// Destructor

LRMachine::~LRMachine()
{
	delete[] m_weights;
}

/////////////////////////////////////////////////////////////////////////
// Resize

bool LRMachine::resize(int size)
{
	if (size < 1)
	{
		return false;
	}

	if (m_size != size)
	{
		m_size = size;
		delete[] m_weights;
		m_weights = new double[m_size + 1];
		for (int i = 0; i <= m_size; i ++)
		{
			m_weights[i] = 0.0;
		}
	}

	return true;
}

/////////////////////////////////////////////////////////////////////////
// Set machine's parameters

void LRMachine::setThreshold(double threshold)
{
	m_threshold = threshold;
}

void LRMachine::setWeights(const double* weights)
{
	for (int i = 0; i <= m_size; i ++)
	{
		m_weights[i] = weights[i];
	}
}

/////////////////////////////////////////////////////////////////////////
// Process the input tensor

bool LRMachine::forward(const Tensor& input)
{
	// Check input type
	if (	input.getDatatype() != Tensor::Double ||
		input.nDimension() != 1 ||
		input.size(0) != m_size)
	{
		message("LRMachine::forward - invalid input tensor!\n");
		return false;
	}

	// OK
	const double* data = (const double*)input.dataR();
	*m_poutput = sigmoid(data, m_weights, m_size);
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Apply the sigmoid function on some data

double LRMachine::sigmoid(const double* data, const double* weights, int size)
{
	double sum = weights[size];
	for (int i = 0; i < size; i ++)
	{
		sum += data[i] * weights[i];
	}

	return 1.0 / (1.0 + exp(-sum));
}

double LRMachine::sigmoidEps(const double* data, const double* weights, int size, double eps)
{
	double sum = weights[size];
	for (int i = 0; i < size; i ++)
	{
		sum += data[i] * weights[i];
	}

	return getInRange(1.0 / (1.0 + exp(-sum)), eps, 1.0 - eps);
}

/////////////////////////////////////////////////////////////////////////
// Loading/Saving the content from files (\emph{not the options})

bool LRMachine::loadFile(File& file)
{
	// Check the ID
	int id;
	if (file.taggedRead(&id, 1, "ID") != 1)
	{
		Torch::message("LRMachine::load - failed to read <ID> field!\n");
		return false;
	}
	if (id != getID())
	{
		Torch::message("LRMachine::load - invalid <ID>, this is not a LRMachine model!\n");
		return false;
	}

	// Read the size
	int size;
	if (file.taggedRead(&size, 1, "SIZE") != 1)
	{
		Torch::message("LRMachine::load - failed to read <SIZE> field!\n");
		return false;
	}
	if (resize(size) == false)
	{
		Torch::message("LRMachine::load - invalid <SIZE> field!\n");
		return false;
	}

	// Read the weights
	if (file.taggedRead(m_weights, m_size + 1, "WEIGHTS") != m_size + 1)
	{
		Torch::message("LRMachine::load - failed to read <WEIGHTS> field!\n");
		return false;
	}

	// Read the threshold
	if (file.taggedRead(&m_threshold, 1, "THRESHOLD") != 1)
	{
		Torch::message("LRMachine::load - failed to read <THRESHOLD> field!\n");
		return false;
	}

	// OK
	return true;
}

bool LRMachine::saveFile(File& file) const
{
	// Write the ID
	int id = getID();
	if (file.taggedWrite(&id, 1, "ID") != 1)
	{
		Torch::message("LRMachine::save - failed to write <ID> field!\n");
		return false;
	}

	// Write the size
	if (file.taggedWrite(&m_size, 1, "SIZE") != 1)
	{
		Torch::message("LRMachine::save - failed to write <SIZE> field!\n");
		return false;
	}

	// Write the weights
	if (file.taggedWrite(m_weights, m_size + 1, "WEIGHTS") != m_size + 1)
	{
		Torch::message("LRMachine::save - failed to write <WEIGHTS> field!\n");
		return false;
	}

	// Write the threshold
	if (file.taggedWrite(&m_threshold, 1, "THRESHOLD") != 1)
	{
		Torch::message("LRMachine::save - failed to write <THRESHOLD> field!\n");
		return false;
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////

}
