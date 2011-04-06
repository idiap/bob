#ifndef _TORCHVISION_SCANNING_SELECTOR_H_
#define _TORCHVISION_SCANNING_SELECTOR_H_

#include "core/Object.h"		// <Selector> is a <Torch::Object>
#include "scanning/Pattern.h"		// works on <Pattern>s

namespace Torch
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::Scanning::Selector
	//	- given a list of candidate sub-windows,
	//		it will select just some of these sub-windows as the best ones!
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class Selector : public Torch::Object
	{
	public:

		// Constructor
		Selector();

		// Destructor
		virtual ~Selector();

		// Delete all stored patterns
		virtual void			clear() { m_patterns.clear(); }

		// Process the list of candidate sub-windows and select the best ones
		// (this will accumulate them to the pattern list)
		virtual bool			process(const PatternList& candidates) = 0;

		// Return the result
		const PatternList&		getPatterns() const { return m_patterns; }

	protected:

		/////////////////////////////////////////////////////////////////
		// Attributes

		// Result: best pattern selected from the candidates (the 4D scanning space)
		PatternList			m_patterns;
	};

	/////////////////////////////////////////////////////////////////////////
}

#endif
