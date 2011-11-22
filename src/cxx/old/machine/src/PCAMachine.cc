/**
 * @file cxx/old/machine/src/PCAMachine.cc
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
#include "machine/PCAMachine.h"
#include "core/File.h"

namespace Torch {

PCAMachine::PCAMachine() : EigenMachine(), Xm(0)
{
	m_parameters->addDarray("Xm", 0, 0.0,"Mean vector");
	
}

PCAMachine::PCAMachine(int n_inputs_) : EigenMachine(n_inputs_)
{
	m_parameters->addDarray("Xm", n_inputs_, 0.0,"Mean vector");

	init_();
	resize(n_inputs_);
}


bool PCAMachine::init_()
{
	EigenMachine::init_();
	Xm = m_parameters->getDarray("Xm");

	return true;
}

bool PCAMachine::forward(const Tensor& input)
{
	// Accept only tensors of Double
	if( input.getDatatype() != Tensor::Double)
	{
		warning("PCAMachine::forward() : incorrect tensor type.");

		return false;
	}

	if( input.size(0) != n_inputs)
	{
		warning("PCAMachine::forward() : incorrect input size along dimension 0 (%d != %d).", input.size(0), n_inputs);

		return false;
	}


	// If the tensor is 1D then considers it as a vector
	if (    input.nDimension() == 1)
	{
		DoubleTensor *t_input = (DoubleTensor *) &input;

		resize(n_outputs);

    const double *src = (const double *) t_input->dataR();
    double *dst = (double *) m_output.dataW();

		double *src_centered = new double[n_inputs];
		for( int j = 0; j < n_inputs ; j++ )
		{
			src_centered[j] = src[j] - Xm[j];
//message("j: %d src: %f Xm: %f -> src_centered: %f", j , src[j], Xm[j], src_centered[j] );
		}
	
/*		for( int i = 0 ; i < n_outputs ; i++)
		{
			for( int j = 0 ; j < n_inputs ; j++)
				print(" eigenvectors[%d,%d]: %f",i, j, eigenvectors[j*n_inputs + i] );
			print("\n");
		}
*/				
		
		for(int i = 0 ; i < n_outputs ; i++)
		{
				dst[i] = 0.;
				for( int j = 0 ; j < n_inputs ; j++)
					dst[i] += eigenvectors[j*n_inputs + i] * src_centered[j];
		}
		delete [] src_centered;
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

			double *src_centered = new double[n_inputs];
			for( int j = 0; j < n_inputs ; j++ )
				src_centered[j] = src[j] - Xm[j];
		
      for(int i = 0 ; i < n_outputs ; i++)
      {
				dst[i] = 0.; 
				for( int j = 0 ; j < n_inputs ; j++)
					dst[i] += eigenvectors[j*n_inputs + i] * src_centered[j];
      }
			delete [] src_centered;
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

        double *src_centered = new double[n_inputs];
        for( int j = 0; j < n_inputs ; j++ )
          src_centered[j] = src[j] - Xm[j];

        for(int i = 0 ; i < n_outputs ; i++)
        {
          dst[i] = 0.; 
          for( int j = 0 ; j < n_inputs ; j++)
            dst[i] += eigenvectors[j*n_inputs + i] * src_centered[j];
        }
        delete [] src_centered;
      }
    }
  }
  else
  {
    warning("PCAMachine::forward() : don't know how to deal with %d dimensions sorry :-(", input.nDimension());

    return false;
  }

  return true;
}

// TODO: Check following definitions
bool PCAMachine::loadFile(File& file)
{
  // Check the ID
  int id;
  if (file.taggedRead(&id, 1, "ID") != 1)
  {
		Torch::message("PCAMachine::load - failed to read <ID> field!\n");
		return false;
	}
	if (id != getID())
	{
		Torch::message("PCAMachine::load - invalid <ID>, this is not a PCAMachine Machine!\n");
		return false;
	}

	if(m_parameters->loadFile(file) == false)
	{
		Torch::message("PCAMachine::load - failed to load parameters\n");
		return false;
	}

  n_outputs = m_parameters->getI("n_outputs");

	init_();

	resize( n_outputs);

	return true;
}

bool PCAMachine::saveFile(File& file) const
{
	// Write the machine ID
	const int id = getID();
	if (file.taggedWrite(&id, 1, "ID") != 1)
	{
		Torch::message("PCAMachine::save - failed to write <ID> field!\n");
		return false;
	}

	if(m_parameters->saveFile(file) == false)
	{
		Torch::message("PCAMachine::save - failed to write parameters\n");
		return false;
	}

	// OK
	return true;
}

PCAMachine::~PCAMachine()
{
}

}

