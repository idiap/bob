#ifndef _TORCHVISION_SCANNING_IP_SW_EVALUATOR_H_
#define _TORCHVISION_SCANNING_IP_SW_EVALUATOR_H_

#include "ipSubWindow.h"		// <ipSWEvaluator> is a <ipSubWindow>

namespace Torch
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::ipSWEvaluator
	//	- use some model to check if some sub-window contains a pattern
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class ipSWEvaluator : public ipSubWindow
	{
	public:

		// Constructor
		ipSWEvaluator();

		// Destructor
		virtual ~ipSWEvaluator();

		/////////////////////////////////////////////////////////////////
		// Access functions

		// Get the result - the sub-window contains the pattern?!
		bool			isPattern() const { return m_isPattern; }
		//	... get the model confidence of this
		float			getConfidence() const { return m_confidence; }

		// Get the model size/threshold
		virtual int		getModelWidth() const = 0;
		virtual int		getModelHeight() const = 0;
		virtual float		getModelThreshold() const = 0;

	protected:

		/////////////////////////////////////////////////////////////////
		// Attributes

		// Model confidence (about finding a pattern in the given sub-window)
		bool			m_isPattern;
		float			m_confidence;
	};
}

#endif
