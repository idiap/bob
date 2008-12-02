#include "DummySelector.h"

namespace Torch
{

/////////////////////////////////////////////////////////////////////////
// Constructor

DummySelector::DummySelector()
{
}

/////////////////////////////////////////////////////////////////////////
// Destructor

DummySelector::~DummySelector()
{
}

/////////////////////////////////////////////////////////////////////////
// Delete all stored patterns

void DummySelector::clear()
{
	Selector::clear();
}

/////////////////////////////////////////////////////////////////////////
// Process the list of candidate sub-windows and select the best ones
// (this will accumulate them to the pattern list)

bool DummySelector::process(const PatternSpace& candidates)
{
        const PatternList& pattList = candidates.getPatternList();
        const int no_patterns = pattList.size();

	// Just copy each sub-window from the pattern space to the output list
	m_patterns.clear();
	for (int i = 0; i < no_patterns; i ++)
	{
	        m_patterns.add(pattList.get(i));
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////

}

