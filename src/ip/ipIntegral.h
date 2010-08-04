#ifndef _TORCHVISION_IP_INTEGRAL_H_
#define _TORCHVISION_IP_INTEGRAL_H_

#include "core/ipCore.h"		// <ipIntegral> is a <Torch::ipCore>

namespace Torch
{
        class Tensor;

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
	//		|XXX|   ---->  |  ipIntegral  | ---->   |XXX|
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

                ////////////////////////////////////////////////////////////////
                // Operator to modify the pixel value that it is to be stored
                //      by the integral image
                //
                //      II(x,y) = \sum_{i=1}^{x-1} \sum_{j=1}^{y-1} PixelOperator(I(i,j))
                ////////////////////////////////////////////////////////////////
                struct PixelOperator
                {
                public:
                        // Constructor
                        PixelOperator() { }

                        // Destructor
                        virtual ~PixelOperator() { }

                        // Process some pixel - should be overriden
                        virtual int     compute(char px) { return px; }
                        virtual int     compute(short px) { return px; }
                        virtual int     compute(int px) { return px; }
                        virtual long    compute(long px) { return px; }
                        virtual double  compute(float px) { return px; }
                        virtual double  compute(double px) { return px; }
                };
                ////////////////////////////////////////////////////////////////

		// Constructor
		ipIntegral();

		// Destructor
		virtual ~ipIntegral();

		// Set the pixel (NULL means the actual pixel value will be used)
		bool            setPixelOperator(PixelOperator* pixelOp);

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

		PixelOperator*  m_pixelOp;
	};
}

#endif
