#ifndef _TORCH5SPRO_MSE_CRITERION_H_
#define _TORCH5SPRO_MSE_CRITERION_H_

#include "machine/Criterion.h"

namespace Torch
{
        //////////////////////////////////////////////////////////////////////////////////////
	// Torch::MSECriterion:
	//
	// TODO: doxygen header!
	//////////////////////////////////////////////////////////////////////////////////////

	class MSECriterion : public Criterion
	{
	public:

		/// Constructor
		MSECriterion(const int target_size);
		
		/// Destructor
		virtual ~MSECriterion();

		///////////////////////////////////////////////////////////

		///
		virtual bool 	forward(const DoubleTensor *machine_output, const Tensor *target);

		///////////////////////////////////////////////////////////
	};

}

#endif
