#ifndef _TORCHVISION_SCANNING_IP_SW_PRUNER_H_
#define _TORCHVISION_SCANNING_IP_SW_PRUNER_H_

#include "core/ipCore.h"		// <ipSWPruner> is an <ipCore>

namespace Torch
{
   	/////////////////////////////////////////////////////////////////////////
	// Torch::ipSWPruner
	//	- rejects some sub-window (e.g. based on the pixel/edge variance
	//		- too smooth or too noisy)
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class ipSWPruner : public ipCore
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
