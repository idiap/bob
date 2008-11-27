#ifndef _TORCHVISION_IP_INTEGRAL_H_
#define _TORCHVISION_IP_INTEGRAL_H_

#include "ipCore.h"		// <ipIntegral> is a <Torch::ipCore>

namespace Torch
{
        class CharTensor;
        class ShortTensor;
        class IntTensor;
        class LongTensor;
        class FloatTensor;
        class DoubleTensor;

	/////////////////////////////////////////////////////////////////////////
	// Torch::ipIntegral:
	//	This class is designed to compute the integral image of any 2D/3D tensor type:
	//              (height x width [x color channels/modes])
        //      The result will have the same dimensions and size/dimension as the input,
        //              but the input type will vary like:
        //
        //              Input:                          Output:
        //              -----------------------------------------
        //              Char            =>              Int
	//		Short           =>              Int
	//		Int             =>              Int
	//		Long            =>              Long
	//		Float           =>              Double
	//		Double          =>              Double
	//
        //      NB: For a 3D tensor, the integral image is computed for each 3D channel
        //              (that is the third dimension -> e.g. color channels).
        //
	//	\begin{equation}
	//		II(x,y) = \sum_{i=1}^{x-1} \sum_{j=1}^{y-1} I(i,j)
	//	\end{equation}
	//
    	//	\begin{verbatim}
        //		+---+          +--------------+         +---+
	//		|XXX|	       |              |         |XXX|
	//		|XXX|   ---->  |  ipIdentity  | ---->   |XXX|
	//		|XXX|          |              |         |XXX|
	//		+---+          +--------------+         +---+
	//		image                                integral image
	//	\end{verbatim}
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class ipIntegral : public ipCore
	{
	public:

		// Constructor
		ipIntegral();

		// Destructor
		virtual ~ipIntegral();

	protected:

		//////////////////////////////////////////////////////////

		/// Check if the input tensor has the right dimensions and type - overriden
		virtual bool	checkInput(const Tensor& input) const;

		/// Allocate (if needed) the output tensors given the input tensor dimensions - overriden
		virtual bool	allocateOutput(const Tensor& input);

		/// Process some input tensor (the input is checked, the outputs are allocated) - overriden
		virtual bool	processInput(const Tensor& input);

		//////////////////////////////////////////////////////////

	private:

                /////////////////////////////////////////////////////////////////
		// Attributes

		//
	};
}

#endif
