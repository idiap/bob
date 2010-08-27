#include "machine/MeanVarNorm.h"
#include "core/File.h"

namespace Torch {

//////////////////////////////////////////////////////////////////////////
// Constructor

MeanVarNorm::MeanVarNorm()
{
   	n_inputs = 0;
	m_mean = NULL;
	m_stdv = NULL;

	frame_in_ = new DoubleTensor();
	sequence_in_ = new DoubleTensor();
	frame_out_ = new DoubleTensor();
	sequence_out_ = new DoubleTensor();
}

MeanVarNorm::MeanVarNorm(const int n_inputs_, DataSet *dataset)
{
	m_mean = NULL;
	m_stdv = NULL;

	frame_in_ = new DoubleTensor();
	sequence_in_ = new DoubleTensor();
	frame_out_ = new DoubleTensor();
	sequence_out_ = new DoubleTensor();

	init_(n_inputs_);

	Torch::print("MeanVarNorm() Number of examples in the DataSet %d\n", dataset->getNoExamples());

	int T = 0;
	for(long t = 0 ; t < dataset->getNoExamples() ; t++)
	{
		Tensor *example = dataset->getExample(t);

		if (	example->getDatatype() != Tensor::Double)
		{
			warning("MeanVarNorm() : incorrect tensor type.");
			break;
		}

		if (example->size(0) != n_inputs)
		{
			warning("MeanVarNorm() : incorrect input size along dimension 0 (%d != %d).", example->size(0), n_inputs);
			break;
		}

		// If the tensor is 1D then considers it as a vector
		if (example->nDimension() == 1)
		{
			DoubleTensor *t_input = (DoubleTensor *) example;

			//Torch::print("MeanVarNorm() processing a frames of size %d\n", n_inputs);

			for(int i = 0 ; i < n_inputs ; i++)
			{
			   	double z = (*t_input)(i);
				m_mean[i] += z;
				m_stdv[i] += z*z;
			}
			
			T++;
		}
		// If the tensor is 2D then considers it as a sequence along the first dimension
		else if (example->nDimension() == 2)
		{
			int n_frames_per_sequence = example->size(1);

			//Torch::print("MeanVarNorm() processing a sequence of %d frames of size %d\n", n_frames_per_sequence, n_inputs);

			for(int f = 0 ; f < n_frames_per_sequence ; f++)
			{
				frame_in_->select(example, 1, f);

				for(int i = 0 ; i < n_inputs ; i++)
				{
			   		double z = (*frame_in_)(i);
					m_mean[i] += z;
					m_stdv[i] += z*z;
				}
		
			}

			T += n_frames_per_sequence;
		}
		// If the tensor is 3D then considers it as a sequence of sequence along the first dimension
		else if (example->nDimension() == 3)
		{
			int n_sequences_per_sequence = example->size(2);
			int n_frames_per_sequence = example->size(1);

			//Torch::print("MeanVarNorm() processing a sequence of %d sequences of %d frames of size %d\n", n_sequences_per_sequence, n_frames_per_sequence, n_inputs);

			for(int s = 0 ; s < n_sequences_per_sequence ; s++)
			{
				sequence_in_->select(example, 2, s);
				
				for(int f = 0 ; f < n_frames_per_sequence ; f++)
				{
					frame_in_->select(sequence_in_, 1, f);

					for(int i = 0 ; i < n_inputs ; i++)
					{
			   			double z = (*frame_in_)(i);
						m_mean[i] += z;
						m_stdv[i] += z*z;
					}
				}
			}

			T += n_frames_per_sequence * n_sequences_per_sequence;
		}
		else
		{
			warning("MeanVarNorm() : don't know how to deal with %d dimensions sorry :-(", example->nDimension());
			break;
		}
	}

	Torch::print("MeanVarNorm() Total number of frames %d\n", T);

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

	//for(int i = 0; i < n_inputs; i++) Torch::print(" mean[%d] = %g \t stdv[%d] = %g\n", i, m_mean[i], i, m_stdv[i]);
}

void MeanVarNorm::init_(const int n_inputs_)
{
	n_inputs = n_inputs_;

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
		
bool MeanVarNorm::resize(const int n_inputs_)
{
	m_output.resize(n_inputs_);

	return true;
}
		
bool MeanVarNorm::resize(const int n_inputs_, const int n_frames_per_sequence_)
{
	m_output.resize(n_inputs_, n_frames_per_sequence_);

	return true;
}

bool MeanVarNorm::resize(const int n_inputs_, const int n_frames_per_sequence_, const int n_sequences_per_sequence_)
{
	m_output.resize(n_inputs_, n_frames_per_sequence_, n_sequences_per_sequence_);

	return true;
}

//////////////////////////////////////////////////////////////////////////
// Destructor

MeanVarNorm::~MeanVarNorm()
{
	delete frame_in_;
	delete sequence_in_;
	delete frame_out_;
	delete sequence_out_;

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
	// Accept only tensors of Double
	if (	input.getDatatype() != Tensor::Double)
	{
		warning("MeanVarNorm::forward() : incorrect tensor type.");

		return false;
	}
	if (	input.size(0) != n_inputs)
	{
		warning("MeanVarNorm::forward() : incorrect input size along dimension 0 (%d != %d).", input.size(0), n_inputs);

		return false;
	}

	// If the tensor is 1D then considers it as a vector
	if (input.nDimension() == 1)
	{
		DoubleTensor *t_input = (DoubleTensor *) &input;

		//Torch::print("MeanVarNorm::forward() processing a frame of size %d\n", n_inputs);
	   	resize(n_inputs);

		for(int i = 0 ; i < n_inputs ; i++)
			m_output(i) = ((*t_input)(i) - m_mean[i]) / m_stdv[i];
	}
	// If the tensor is 2D then considers it as a sequence along the first dimension
	else if (input.nDimension() == 2)
	{
		DoubleTensor *t_input = (DoubleTensor *) &input;
		DoubleTensor *t_output = (DoubleTensor *) &m_output;
		int n_frames_per_sequence = t_input->size(1);

		//Torch::print("MeanVarNorm::forward() processing a sequence of %d frames of size %d\n", n_frames_per_sequence, n_inputs);
	   	resize(n_inputs, n_frames_per_sequence);

		for(int f = 0 ; f < n_frames_per_sequence ; f++)
		{
			frame_in_->select(t_input, 1, f);

			frame_out_->select(t_output, 1, f);

			for(int i = 0 ; i < n_inputs ; i++)
				(*frame_out_)(i) = ((*frame_in_)(i) - m_mean[i]) / m_stdv[i];
		}
	}
	// If the tensor is 3D then considers it as a sequence of sequence along the first dimension
	else if (input.nDimension() == 3)
	{
		DoubleTensor *t_input = (DoubleTensor *) &input;
		DoubleTensor *t_output = (DoubleTensor *) &m_output;
		int n_sequences_per_sequence = t_input->size(2);
		int n_frames_per_sequence = t_input->size(1);

		//Torch::print("MeanVarNorm::forward() processing a sequence of %d sequences of %d frames of size %d\n", n_sequences_per_sequence, n_frames_per_sequence, n_inputs);
	   	resize(n_inputs, n_frames_per_sequence, n_sequences_per_sequence);

		for(int s = 0 ; s < n_sequences_per_sequence ; s++)
		{
			sequence_in_->select(t_input, 2, s);
			sequence_out_->select(t_output, 2, s);
			
			for(int f = 0 ; f < n_frames_per_sequence ; f++)
			{
				frame_in_->select(sequence_in_, 1, f);

				frame_out_->select(sequence_out_, 1, f);

				for(int i = 0 ; i < n_inputs ; i++)
					(*frame_out_)(i) = ((*frame_in_)(i) - m_mean[i]) / m_stdv[i];
			}
		}
	}
	else
	{
		warning("MeanVarNorm::forward() : don't know how to deal with %d dimensions sorry :-(", input.nDimension());

		return false;
	}

	return true;
}

///////////////////////////////////////////////////////////
// Loading/Saving the content from files (\emph{not the options}) - overriden

bool MeanVarNorm::loadFile(File& file)
{
	// Check the ID
	int id;
	if (file.taggedRead(&id, 1, "ID") != 1)
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

	if (file.taggedRead(&n_inputs_, 1, "N_INPUTS") != 1)
	{
		Torch::message("MeanVarNorm::load - failed to read <N_INPUTS> field!\n");
		return false;
	}

	init_(n_inputs_);

	// Read the machine parameters
	const int ret1 = file.taggedRead(m_mean, n_inputs, "MEAN");
	if (ret1 != n_inputs)
	{
	        Torch::message("MeanVarNorm::load - failed to read <MEAN> field!\n");
		return false;
	}

	// Read the machine parameters
	const int ret2 = file.taggedRead(m_stdv, n_inputs, "STDV");
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
	if (file.taggedWrite(&id, 1, "ID") != 1)
	{
		Torch::message("MeanVarNorm::save - failed to write <ID> field!\n");
		return false;
	}

	// Write the input size
	if (file.taggedWrite(&n_inputs, 1, "N_INPUTS") != 1)
	{
		Torch::message("MeanVarNorm::save - failed to write <N_INPUTS> field!\n");
		return false;
	}

	// Write the mean
	if (file.taggedWrite(m_mean, n_inputs, "MEAN") != n_inputs)
	{
		Torch::message("MeanVarNorm::save - failed to write <MEAN> field!\n");
		return false;
	}

	// Write the stdv
	if (file.taggedWrite(m_stdv, n_inputs, "STDV") != n_inputs)
	{
		Torch::message("MeanVarNorm::save - failed to write <STDV> field!\n");
		return false;
	}

	// OK
	return true;
}

//////////////////////////////////////////////////////////////////////////
}
