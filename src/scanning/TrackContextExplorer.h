#ifndef _TORCHVISION_SCANNING_TRACK_CONTEXT_EXPLORER_H_
#define _TORCHVISION_SCANNING_TRACK_CONTEXT_EXPLORER_H_

#include "ContextExplorer.h"		// <TrackContextExplorer> is a <ContextExplorer>

namespace Torch
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::TrackContextExplorer
	//	- process only a specified target sub-window
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class TrackContextExplorer : public ContextExplorer
	{
	public:

		// Constructor
		TrackContextExplorer(Mode mode = Scanning);

		// Destructor
		virtual ~TrackContextExplorer();
		
		// Change the sub-windows to process
		void setSeedPatterns(const PatternList& patterns);
		
	protected:
	  
		// Initialize the sub-windows to process
		virtual bool		initContext();

		/////////////////////////////////////////////////////////////////
		// Attributes
	
		PatternList		m_seed_patterns;
	};
}

#endif
