/**
 * @file cxx/old/machine/src/EigenMachine.cc
 * @date Wed Apr 6 14:49:40 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "machine/EigenMachine.h"
#include "core/File.h"

namespace Torch {

//////////////////////////////////////////////////////////////////////////
// Constructor

EigenMachine::EigenMachine(): n_inputs(0), n_outputs(0), eigenvalues(0), eigenvectors(0)
{
	m_parameters->addI("n_inputs", 0, "number of dimensions");
	m_parameters->addI("n_outputs", 0, "number of output dimensions");
  m_parameters->addDarray("eigenvalues", 0, 0.0, "eigenvalues of the covariance matrix");
  m_parameters->addDarray("eigenvectors", 0, 0.0, "eigenvectors of the covariance matrix");

	m_parameters->addD("variance", 1.0, "variance to keep");

  frame_in_ = new DoubleTensor();
  sequence_in_ = new DoubleTensor();
  frame_out_ = new DoubleTensor();
  sequence_out_ = new DoubleTensor();
}

EigenMachine::EigenMachine(const int n_inputs_ ): n_inputs(n_inputs_), n_outputs(n_inputs_), eigenvalues(0), eigenvectors(0) 
{
	m_parameters->addI("n_inputs", n_inputs_, "number of dimensions");
	m_parameters->addI("n_outputs", n_inputs_, "number of output dimensions");
  m_parameters->addDarray("eigenvalues", n_inputs_, 0.0, "eigenvalues of the covariance matrix");
  m_parameters->addDarray("eigenvectors", n_inputs_*n_inputs_, 0.0, "eigenvectors of the covariance matrix");

	m_parameters->addD("variance", 1.0, "variance to keep");

	init_();

	resize(n_inputs_);
  frame_in_ = new DoubleTensor();
  sequence_in_ = new DoubleTensor();
  frame_out_ = new DoubleTensor();
  sequence_out_ = new DoubleTensor();
}

bool EigenMachine::init_()
{
	n_inputs = m_parameters->getI("n_inputs");
	n_outputs = m_parameters->getI("n_outputs");
	
	eigenvalues = m_parameters->getDarray("eigenvalues");
	eigenvectors = m_parameters->getDarray("eigenvectors");

	variance = m_parameters->getI("variance");

	return true;
}

bool EigenMachine::setNumberOfRelevantEigenvectors()
{
	if( variance <= 0.0 )
	{
		n_outputs = n_inputs;
	}
	else
	{
		double sum = eigenvalues[0];
		double variance_ = 0.;
		
//		double min_eigenv = eigenvalues[0];
//		double max_eigenv = eigenvalues[0];
		
		for(int i = 1 ; i < n_inputs ; i++)
		{
			double val = eigenvalues[i];
//			if(val < min_eigenv) min_eigenv = val;
//			if(val > max_eigenv) max_eigenv = val;
    
      sum += val;
    }

		int i = 0;
		while (variance_ < (sum*variance))
		{
			variance_ += eigenvalues[i];
			i++;
		}

		n_outputs = i;
	}
	return true;
}

bool EigenMachine::resize(const int n_outputs_)
{
        m_output.resize(n_outputs_);
        return true;
}
    
bool EigenMachine::resize(const int n_outputs_, const int n_frames_per_sequence_)
{
        m_output.resize(n_outputs_, n_frames_per_sequence_);
        return true;
}

bool EigenMachine::resize(const int n_outputs_, const int n_frames_per_sequence_, const int n_sequences_per_sequence_)
{
        m_output.resize(n_outputs_, n_frames_per_sequence_, n_sequences_per_sequence_);
        return true;
}
	

//////////////////////////////////////////////////////////////////////////
// Destructor
EigenMachine::~EigenMachine()
{
        delete frame_in_;
        delete sequence_in_;
        delete frame_out_;
        delete sequence_out_;
}

bool EigenMachine::forward(const Tensor& input)
{
	// Accept only tensors of Double
	if (	input.getDatatype() != Tensor::Double)
	{
		warning("EigenMachine::forward() : incorrect tensor type.");
		return false;
	}

	// Check size of the first dimension
	if (	input.size(0) != n_inputs)
	{
		warning("EigenMachine::forward() : incorrect input size along dimension 0 (%d != %d).", input.size(0), n_inputs);
		return false;
	}


	// If the tensor is 1D then considers it as a vector
	if (    input.nDimension() == 1)
	{
		DoubleTensor *t_input = (DoubleTensor *) &input;
		resize(n_outputs);

    const double *src = (const double *) t_input->dataR();
    double *dst = (double *) m_output.dataW();

		for(int i = 0 ; i < n_outputs ; i++)
		{
			dst[i] = 0.;
			for( int j = 0 ; j < n_inputs ; j++)
				dst[i] += eigenvectors[j*n_inputs + i] * src[j];
		}
	}
	// If the tensor is 2D then considers it as a sequence along the first dimension
	else if (input.nDimension() == 2)
	{
		DoubleTensor *t_input = (DoubleTensor *) &input;
		DoubleTensor *t_output = (DoubleTensor *) &m_output;
		
		int n_frames_per_sequence = t_input->size(1);

   	resize(n_outputs, n_frames_per_sequence );

		for(int f = 0 ; f < n_frames_per_sequence ; f++)
		{
			frame_in_->select(t_input, 1, f); 
			const double *src = (const double *) frame_in_->dataR();

			frame_out_->select(t_output, 1, f); 
			double *dst = (double *) frame_out_->dataW();

      for(int i = 0 ; i < n_outputs ; i++)
      {
				dst[i] = 0.; 
				for( int j = 0 ; j < n_inputs ; j++)
					dst[i] += eigenvectors[j*n_inputs + i] * src[j];
	    }
		}
	}
	// If the tensor is 3D then considers it as a sequence of sequence along the first dimension
	else if (input.nDimension() == 3)
	{
		DoubleTensor *t_input = (DoubleTensor *) &input;
		DoubleTensor *t_output = (DoubleTensor *) &m_output;

		int n_sequences_per_sequence = t_input->size(2);
		int n_frames_per_sequence = t_input->size(1);

		resize(n_outputs, n_frames_per_sequence, n_sequences_per_sequence);

		for(int s = 0 ; s < n_sequences_per_sequence ; s++)
		{
      sequence_in_->select(t_input, 2, s);
      sequence_out_->select(t_output, 2, s);

      for(int f = 0 ; f < n_frames_per_sequence ; f++)
      {
        frame_in_->select(sequence_in_, 1, f);
        double *src = (double *) frame_in_->dataR();

        frame_out_->select(sequence_out_, 1, f);
        double *dst = (double *) frame_out_->dataW();

	      for(int i = 0 ; i < n_outputs ; i++)
	      {
          dst[i] = 0.; 
          for( int j = 0 ; j < n_inputs ; j++)
            dst[i] += eigenvectors[j*n_inputs + i] * src[j];
	      }
      }
    }
  }
	else
	{
		warning("EigenMachine::forward() : don't know how to deal with %d dimensions sorry :-(", input.nDimension());
		return false;
	}

	return true;
}

///////////////////////////////////////////////////////////
// Loading/Saving the content from files (\emph{not the options}) - overriden

bool EigenMachine::loadFile(File& file)
{
	// Check the ID
	int id;
	if (file.taggedRead(&id, 1, "ID") != 1)
	{
		Torch::message("EigenMachine::load - failed to read <ID> field!\n");
		return false;
	}
	if (id != getID())
	{
		Torch::message("EigenMachine::load - invalid <ID>, this is not a EigenMachine Machine!\n");
		return false;
	}

	if(m_parameters->loadFile(file) == false)
	{
		Torch::message("EigenMachine::load - failed to load parameters\n");
		return false;
	}

  n_outputs = m_parameters->getI("n_outputs");

	init_();
	resize( n_outputs);

	// OK
	return true;
}

bool EigenMachine::saveFile(File& file) const
{
	// Write the machine ID
	const int id = getID();
	if (file.taggedWrite(&id, 1, "ID") != 1)
	{
		Torch::message("EigenMachine::save - failed to write <ID> field!\n");
		return false;
	}

	if(m_parameters->saveFile(file) == false)
	{
		Torch::message("EigenMachine::save - failed to write parameters\n");
		return false;
	}

	// OK
	return true;
}

//////////////////////////////////////////////////////////////////////////
}

