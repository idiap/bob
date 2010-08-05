#ifndef _TORCH5SPRO_TWO_CLASS_NLL_CRITERION_H_
#define _TORCH5SPRO_TWO_CLASS_NLL_CRITERION_H_

#include "Criterion.h"

namespace Torch
{
        //////////////////////////////////////////////////////////////////////////////////////
	// Torch::TwoClassNLLCriterion:
	//
	// TODO: doxygen header!
	//////////////////////////////////////////////////////////////////////////////////////

	class TwoClassNLLCriterion : public Criterion
	{
	public:

		/// Constructor
		TwoClassNLLCriterion(const double cst = 0.0);
		
		/// Destructor
		virtual ~TwoClassNLLCriterion();

		///////////////////////////////////////////////////////////

		///
		virtual bool 	forward(const DoubleTensor *machine_output, const Tensor *target);

		///////////////////////////////////////////////////////////

		double m_cst;
	};

}

#endif
