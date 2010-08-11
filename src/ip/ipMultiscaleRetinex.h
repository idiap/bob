#ifndef _TORCH5SPRO_IP_MULTISCALE_RETINEX_H_
#define _TORCH5SPRO_IP_MULTISCALE_RETINEX_H_

#include "ip/ipCore.h"

namespace Torch {

	/** This class is designed to perform the Multiscale Retinex algorithm on an image

	    \verbatim

	         +---+          +---------------------+         +---+
	         |xxx|	        |                     |         |XXX|
		 |xxx|   ---->  | ipMultiscaleRetinex | ---->   |XXX|
	         |xxx|          |                     |         |XXX|
	         +---+          +---------------------+         +---+
	
	    \endverbatim

	    @author Guillaume Heusch (heusch@idiap.ch)	    
	    @author Laurent El Shafey (laurent.el-shafey@idiap.ch)
	    
	*/

	class ipMultiscaleRetinex : public ipCore
	{
	public:

		/// constructor
	    	ipMultiscaleRetinex();

		/// destructor
		virtual ~ipMultiscaleRetinex();
	
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

