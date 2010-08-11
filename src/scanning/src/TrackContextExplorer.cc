#include "scanning/TrackContextExplorer.h"

namespace Torch
{

/////////////////////////////////////////////////////////////////////////
// Constructor

TrackContextExplorer::TrackContextExplorer(Mode mode)
	:	ContextExplorer(mode)
{
}

/////////////////////////////////////////////////////////////////////////
// Destructor

TrackContextExplorer::~TrackContextExplorer()
{
}

/////////////////////////////////////////////////////////////////////////
// Change the sub-windows to process

void TrackContextExplorer::setSeedPatterns(const PatternList& patterns)
{
	m_seed_patterns.clear();
	m_seed_patterns.add(patterns);
}

/////////////////////////////////////////////////////////////////////////
// Initialize the sub-windows to process

bool TrackContextExplorer::initContext()
{
	if (m_seed_patterns.isEmpty())
	{
	      return ContextExplorer::initContext();
	}
	else
	{
	      m_data->clear();
	      for (int i = 0; i < m_seed_patterns.size(); i ++)
	      {
		      m_data->storePattern(m_seed_patterns.get(i));
	      }
	      return true;
	}
}

/////////////////////////////////////////////////////////////////////////

}
