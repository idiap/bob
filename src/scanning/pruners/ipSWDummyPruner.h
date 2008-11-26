#ifndef _TORCHVISION_SCANNING_IP_SW_DUMMY_PRUNER_H_
#define _TORCHVISION_SCANNING_IP_SW_DUMMY_PRUNER_H_

#include "ipSWPruner.h"		// <ipSWDummyPruner> is a <ipSWPruner>

namespace Torch
{
	class Tensor;

	/////////////////////////////////////////////////////////////////////////
	// Torch::ipSWDummyPruner
	//	- used for testing
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class ipSWDummyPruner : public ipSWPruner
	{
	public:

		// Different types of dummy pruners
		enum Type
		{
			RejectAll,
			RejectNone,
			RejectRandom
		};

		// Constructor
		ipSWDummyPruner(Type type);

		// Destructor
		virtual ~ipSWDummyPruner();

	protected:

		//////////////////////////////////////////////////////////

		/// Check if the input tensor has the right dimensions and type
		virtual bool		checkInput(const Tensor& input) const;

		/// Allocate (if needed) the output tensors given the input tensor dimensions
		virtual bool		allocateOutput(const Tensor& input);

		/// Process some input tensor (the input is checked, the outputs are allocated)
		virtual bool		processInput(const Tensor& input);

		/////////////////////////////////////////////////////////////////
		// Attributes

		Type			m_type;
	};
}

#endif
