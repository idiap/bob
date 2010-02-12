#ifndef _TORCH5SPRO_IP_SELF_QUOTIENT_IMAGE_H_
#define _TORCH5SPRO_IP_SELF_QUOTIENT_IMAGE_H_

#include "ipCore.h"

namespace Torch {

	/** This class is designed to perform the Self Quotient Image algorithm on an image

	    \begin{verbatim}

	         +---+          +---------------------+         +---+
	         |xxx|	        |                     |         |XXX|
		 |xxx|   ---->  | ipSelfQuotientImage | ---->   |XXX|
	         |xxx|          |                     |         |XXX|
	         +---+          +---------------------+         +---+
	
	    \end{verbatim}

	    @author Guillaume Heusch (heusch@idiap.ch)	    
	    @author Laurent El Shafey (laurent.el-shafey@idiap.ch)
	    
	*/

	class ipSelfQuotientImage : public ipCore
	{
	public:

		/// constructor
	    	ipSelfQuotientImage();

		/// destructor
		virtual ~ipSelfQuotientImage();
	
	protected:

		////////////////////////////////////////////////////////////////////////////////////////////

		/// Check if the input tensor has the right dimensions and type - overriden
		virtual bool            checkInput(const Tensor& input) const;
		
		/// Allocate (if needed) the output tensors given the input tensor dimensions - overriden
		virtual bool            allocateOutput(const Tensor& input);
		
		/// Process some input tensor (the input is checked, the outputs are allocated) - overriden
		virtual bool            processInput(const Tensor& input);

		////////////////////////////////////////////////////////////////////////////////////////////
		
	private:

	};

}

#endif

