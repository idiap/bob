#ifndef _TORCHVISION_SCANNING_IP_SW_DUMMY_EVALUATOR_H_
#define _TORCHVISION_SCANNING_IP_SW_DUMMY_EVALUATOR_H_

#include "ipSWEvaluator.h"		// <ipSWDummyEvaluator> is a <ipSWEvaluator>

namespace Torch
{
   	/////////////////////////////////////////////////////////////////////////
	// Torch::ipSWDummyEvaluator
	//	- used just for testing
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class ipSWDummyEvaluator : public ipSWEvaluator
	{
	public:

		// Different types of dummy evaluators
		enum Type
		{
			PassAll,
			PassNone,
			PassRandom
		};

		// Constructor
		ipSWDummyEvaluator(int modelWidth, int modelHeight, Type type);

		// Destructor
		virtual ~ipSWDummyEvaluator();

		// Get the model size/threshold
		virtual int		getModelWidth() const { return m_modelWidth; }
		virtual int		getModelHeight() const { return m_modelHeight; }
		virtual float		getModelThreshold() const { return m_modelThreshold; }

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

		int			m_modelWidth, m_modelHeight;
		float			m_modelThreshold;
		Type			m_type;
	};
}

#endif
