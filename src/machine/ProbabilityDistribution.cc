#include "ProbabilityDistribution.h"
#include "core/Tensor.h"
#include "core/File.h"

namespace Torch {

//////////////////////////////////////////////////////////////////////////
// Constructor

ProbabilityDistribution::ProbabilityDistribution()
{
	n_inputs = 0;
}

ProbabilityDistribution::ProbabilityDistribution(const int n_inputs_)
{
	n_inputs = n_inputs_;
}

//////////////////////////////////////////////////////////////////////////
// Destructor

ProbabilityDistribution::~ProbabilityDistribution()
{
}

bool ProbabilityDistribution::forward(const Tensor& input)
{
	// Accept only Tensor of Double
	if (	input.getDatatype() != Tensor::Double)
	{
		warning("ProbabilityDistribution::forward() : incorrect tensor type.");
		
		return false;
	}

	// Test the number of dimensions of the tensor and then its size along each dimensions
	if (	input.nDimension() >= 1 && input.nDimension() <= 3)
	{
	   	//
		// If the tensor is 1D then considers it as a vector
		if (	input.nDimension() == 1)
		{
			if (	input.size(0) != n_inputs)
			{
				warning("ProbabilityDistribution::forward() : incorrect input size along dimension 0 (%d != %d).", input.size(0), n_inputs);
				
				return false;
			}
		}

		//
		// If the tensor is 2D/3D then considers it as a sequence. The first dimension should be the dimension of the frame.


	   	if(input.nDimension() == 2)
		{
			//int n_frames = input.size(1);

			if (	input.size(0) != n_inputs)
			{
				warning("ProbabilityDistribution::forward() : incorrect input size along dimension 1 (%d != %d).", input.size(0), n_inputs);
				
				return false;
			}
		}

	   	if(input.nDimension() == 3)
		{
			//int n_frames = input.size(2);

			if (	input.size(0) != n_inputs)
			{
				warning("ProbabilityDistribution::forward() : incorrect input size along dimension 2 (%d != %d).", input.size(0), n_inputs);
				
				return false;
			}
		}

		DoubleTensor *t_input = (DoubleTensor *) &input;

		return forward(t_input);
	}
	else 
	{
		warning("ProbabilityDistribution::forward() : incorrect number of dimensions.");
		
		return false;
	}
}

}
