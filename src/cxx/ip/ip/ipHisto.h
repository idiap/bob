#ifndef _TORCHVISION_IP_HISTO_H_
#define _TORCHVISION_IP_HISTO_H_

#include "ip/ipCore.h"		// <ipHisto> is a <Torch::ipCore>

namespace Torch
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::ipHisto
	//	This class is designed to compute the histogram of some Image (3D ShortTensor).
	//	The result is 2D IntTensor with the dimensions (bin counts, planes)
	//
	//	\verbatim
	//
	//	 +---+          +--------------+         +
	//	 |XXX|	        |              |         |   *    *
	//	 |XXX|   ---->  |    ipHisto   | ---->   | * *    *  *   *
	//	 |XXX|          |              |         |** *** *** * * *
	//	 +---+          +--------------+         +------------------+
	//
	//	 image                                         histogram
	//
	//	\endverbatim
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class ipHisto : public ipCore
	{
	public:

		// Constructor
		ipHisto();

		// Destructor
		virtual ~ipHisto();

	protected:

		//////////////////////////////////////////////////////////

		/// Check if the input tensor has the right dimensions and type - overriden
		virtual bool		checkInput(const Tensor& input) const;

		/// Allocate (if needed) the output tensors given the input tensor dimensions - overriden
		virtual bool		allocateOutput(const Tensor& input);

		/// Process some input tensor (the input is checked, the outputs are allocated) - overriden
		virtual bool		processInput(const Tensor& input);

		//////////////////////////////////////////////////////////

	private:

		/////////////////////////////////////////////////////////////////
		// Attributes

		// -
	};
}

#endif
