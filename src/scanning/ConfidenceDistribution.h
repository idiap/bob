#ifndef _TORCHVISION_SCANNING_CONFIDENCE_DISTRIBUTION_H_
#define _TORCHVISION_SCANNING_CONFIDENCE_DISTRIBUTIONH_

#include "Object.h"		// <ConfidenceDistribution> is a <Torch::Object>
#include "Pattern.h"		// works on <CandidatePattern>s

#include <list>			// usefull 

namespace Torch
{	
namespace Scanning 
{
   	/////////////////////////////////////////////////////////////////////////
	// Torch::Scanning::ConfidenceDistribution
	//	- models the confidence distribution in the location & scale space
	//		of all sub-windows to be scanned
	// 
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class ConfidenceDistribution : public Torch::Object 
	{
	public:

		// Constructor
		ConfidenceDistribution();

		// Destructor
		virtual ~ConfidenceDistribution();
	
	protected:

		/////////////////////////////////////////////////////////////////
		// Attributes

		//
	};
}
}

#endif
