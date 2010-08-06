#ifndef _TORCH5SPRO_PROBABILITY_DISTRIBUTION_MACHINE_H_
#define _TORCH5SPRO_PROBABILITY_DISTRIBUTION_MACHINE_H_

#include "core/Machine.h"	// ProbabilityDistribution is a <Machine>

namespace Torch
{
        //////////////////////////////////////////////////////////////////////////////////////
	// Torch::ProbabilityDistribution:
	//
	// TODO: doxygen header!
	//////////////////////////////////////////////////////////////////////////////////////

	class ProbabilityDistribution : public Machine
	{
	public:
		/// Constructor
		ProbabilityDistribution();

		/// Constructor
		ProbabilityDistribution(const int n_inputs_);
		
		/// Destructor
		virtual ~ProbabilityDistribution();

		///
		virtual bool 		prepare() { return true; };
		
		///
		virtual bool 		EMinit() { return true; };

		///
		virtual bool 		EMaccPosteriors(const DoubleTensor *input, const double input_posterior) { return true; };

		///
		virtual bool 		EMupdate() { return true; };
		
		///
		virtual bool 		forward(const Tensor& input);

		///
		virtual bool 		forward(const DoubleTensor *input) = 0;

		///
		virtual bool 		print() { return true; };
		
		///
		virtual bool 		shuffle() { return true; };

		///
		int			getNinputs() { return n_inputs; };
		
	protected:
		int n_inputs;
	};

}

#endif
