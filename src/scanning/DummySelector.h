#ifndef _TORCHVISION_SCANNING_DUMMY_SELECTOR_H_
#define _TORCHVISION_SCANNING_DUMMY_SELECTOR_H_

#include "scanning/Selector.h"		// <DummySelector> is a <Selector>

namespace Torch
{
   	/////////////////////////////////////////////////////////////////////////
	// Torch::Scanning::DummySelector
	//	- just for testing purposes
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class DummySelector : public Selector
	{
	public:

		// Constructor
		DummySelector();

		// Destructor
		virtual ~DummySelector();

		// Delete all stored patterns
		virtual void			clear();

		// Process the list of candidate sub-windows and select the best ones
		// (this will accumulate them to the pattern list)
		virtual bool			process(const PatternList& candidates);

	protected:

		/////////////////////////////////////////////////////////////////
		// Attributes

		//
	};
}

#endif
