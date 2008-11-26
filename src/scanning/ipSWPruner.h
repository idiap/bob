#ifndef _TORCHVISION_SCANNING_IP_SW_PRUNER_H_
#define _TORCHVISION_SCANNING_IP_SW_PRUNER_H_

#include "ipSubWindow.h"	// <ipSWPruner> is a <ipSubWindow>

namespace Torch
{
   	/////////////////////////////////////////////////////////////////////////
	// Torch::ipSWPruner
	//	- rejects some sub-window (e.g. based on the pixel variance
	//		- too smooth or too noisy)
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class ipSWPruner : public ipSubWindow
	{
	public:

		// Constructor
		ipSWPruner();

		// Destructor
		virtual ~ipSWPruner();

		// Get the result - the sub-window is rejected?!
		bool			isRejected() const { return m_isRejected; }

	protected:

		/////////////////////////////////////////////////////////////////
		// Attributes

		bool			m_isRejected;
	};
}

#endif
