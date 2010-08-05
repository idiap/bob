#ifndef _TORCH5SPRO_CRITERION_H_
#define _TORCH5SPRO_CRITERION_H_

#include "core/Object.h"
#include "core/Tensor.h"

namespace Torch
{
        //////////////////////////////////////////////////////////////////////////////////////
	// Torch::Criterion:
	//
	// TODO: doxygen header!
	//////////////////////////////////////////////////////////////////////////////////////

	class Criterion : public Object
	{
	public:

		/// Constructor
		Criterion(const int target_size, const int error_size = 1);
		
		/// Destructor
		virtual ~Criterion();

		///////////////////////////////////////////////////////////

		///
		virtual bool 	forward(const DoubleTensor *machine_output, const Tensor *target) = 0;

		///////////////////////////////////////////////////////////

		DoubleTensor*	m_error;
		DoubleTensor*	m_beta;
		DoubleTensor*	m_target;

	protected:

		///////////////////////////////////////////////////////////////
		// Attributes

		int 		m_target_size;
		int 		m_error_size;

	};

}

#endif
