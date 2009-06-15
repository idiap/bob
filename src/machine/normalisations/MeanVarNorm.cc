#include "MeanVarNorm.h"
#include "File.h"

namespace Torch {

//////////////////////////////////////////////////////////////////////////
// Constructor

MeanVarNorm::MeanVarNorm()
{
   	n_inputs = 0;
	n_outputs = 0;
	m_output = NULL;
	m_mean = NULL;
	m_stdv = NULL;
}

MeanVarNorm::MeanVarNorm(const int n_inputs_, DataSet *dataset)
{
	m_output = NULL;
	m_mean = NULL;
	m_stdv = NULL;

	resize(n_inputs_);

	int T = 0;
	for(long t = 0 ; t < dataset->getNoExamples() ; t++)
	{
		Tensor *example = dataset->getExample(t);

		/* Ideally we should iterate on 2D or 3D tensors if sequences or sequences of sequences
		*/
		if (	example->nDimension() != 1 || example->getDatatype() != Tensor::Double)
		{
			warning("MeanVarNorm() : incorrect number of dimensions or type.");
			break;
		}
		if (	example->size(0) != n_inputs)
		{
			warning("MeanVarNorm() : incorrect input size along dimension 0 (%d != %d).", example->size(0), n_inputs);
			break;
		}

		DoubleTensor *t_input = (DoubleTensor *) example;
		double *src = (double *) t_input->dataR();

		for(int i = 0 ; i < n_inputs ; i++)
		{
		   	double z = src[i];
			m_mean[i] += z;
			m_stdv[i] += z*z;
		}

		T++;
	}


	for(int i = 0; i < n_inputs; i++)
	{
		m_mean[i] /= (double) T;
		m_stdv[i] /= (double) T;
		m_stdv[i] -= m_mean[i] * m_mean[i];
		if(m_stdv[i] <= 0)
		{
			warning("MeanVarNorm() : input column %d has a null stdv. Replaced by 1.", i);
			m_stdv[i] = 1.;
		}
		else m_stdv[i] = sqrt(m_stdv[i]);
	}

}

bool MeanVarNorm::resize(const int n_inputs_)
{
	n_inputs = n_inputs_;
	n_outputs = n_inputs;

	// Free
	if(m_output != NULL) 
	{
		delete m_output;
		m_output = NULL;
	}
	// Reallocate
	m_output = new DoubleTensor(n_outputs);

	if(m_mean != NULL)
	{
		delete [] m_mean;
		m_mean = NULL;
	}
	if(m_stdv != NULL)
	{
		delete [] m_stdv;
		m_stdv = NULL;
	}
	m_mean = new double [n_inputs]; 
	m_stdv = new double [n_inputs]; 
	for(int i = 0 ; i < n_inputs ; i++)
	{
		m_mean[i] = 0.0;
		m_stdv[i] = 0.0;
	}
}

//////////////////////////////////////////////////////////////////////////
// Destructor

MeanVarNorm::~MeanVarNorm()
{
	if(m_output != NULL) 
	{
		delete m_output;
		m_output = NULL;
	}
	if(m_mean != NULL)
	{
		delete [] m_mean;
		m_mean = NULL;
	}
	if(m_stdv != NULL)
	{
		delete [] m_stdv;
		m_stdv = NULL;
	}
}

bool MeanVarNorm::forward(const Tensor& input)
{
	// Accept only 1D tensors of Double
	if (	input.nDimension() != 1 || input.getDatatype() != Tensor::Double)
	{
		warning("MeanVarNorm::forward() : incorrect number of dimensions or type.");
		
		return false;
	}
	if (	input.size(0) != n_inputs)
	{
		warning("MeanVarNorm::forward() : incorrect input size along dimension 0 (%d != %d).", input.size(0), n_inputs);
		
		return false;
	}

	DoubleTensor *t_input = (DoubleTensor *) &input;

	double *src = (double *) t_input->dataR();
	double *dst = (double *) m_output->dataW();
	
	for(int i = 0 ; i < n_inputs ; i++)
		dst[i] = (src[i] - m_mean[i]) / m_stdv[i];

	return true;
}

///////////////////////////////////////////////////////////
// Loading/Saving the content from files (\emph{not the options}) - overriden

bool MeanVarNorm::loadFile(File& file)
{
	// Check the ID
	int id;
	if (file.taggedRead(&id, sizeof(int), 1, "ID") != 1)
	{
		Torch::message("MeanVarNorm::load - failed to read <ID> field!\n");
		return false;
	}
	if (id != getID())
	{
		Torch::message("MeanVarNorm::load - invalid <ID>, this is not a MeanVarNorm Machine!\n");
		return false;
	}

	int n_inputs_;

	if (file.taggedRead(&n_inputs_, sizeof(int), 1, "N_INPUTS") != 1)
	{
		Torch::message("MeanVarNorm::load - failed to read <N_INPUTS> field!\n");
		return false;
	}

	resize(n_inputs_);

	// Read the machine parameters
	const int ret1 = file.taggedRead(m_mean, sizeof(double), n_inputs, "MEAN");
	if (ret1 != n_inputs)
	{
	        Torch::message("MeanVarNorm::load - failed to read <MEAN> field!\n");
		return false;
	}

	// Read the machine parameters
	const int ret2 = file.taggedRead(m_stdv, sizeof(double), n_inputs, "STDV");
	if (ret2 != n_inputs)
	{
	        Torch::message("MeanVarNorm::load - failed to read <MEAN> field!\n");
		return false;
	}

	// OK
	return true;
}

bool MeanVarNorm::saveFile(File& file) const
{
	// Write the machine ID
	const int id = getID();
	if (file.taggedWrite(&id, sizeof(int), 1, "ID") != 1)
	{
		Torch::message("MeanVarNorm::save - failed to write <ID> field!\n");
		return false;
	}

	// Write the input size
	if (file.taggedWrite(&n_inputs, sizeof(int), 1, "N_INPUTS") != 1)
	{
		Torch::message("MeanVarNorm::save - failed to write <N_INPUTS> field!\n");
		return false;
	}

	// Write the mean
	if (file.taggedWrite(m_mean, sizeof(double), n_inputs, "MEAN") != n_inputs)
	{
		Torch::message("MeanVarNorm::save - failed to write <MEAN> field!\n");
		return false;
	}

	// Write the stdv
	if (file.taggedWrite(m_stdv, sizeof(double), n_inputs, "STDV") != n_inputs)
	{
		Torch::message("MeanVarNorm::save - failed to write <STDV> field!\n");
		return false;
	}

	// OK
	return true;
}

//////////////////////////////////////////////////////////////////////////
}
