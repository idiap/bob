#ifndef _TORCH5SPRO_IP_HISTO_EQUAL_H_
#define _TORCH5SPRO_IP_HISTO_EQUAL_H_

#include "core/ipCore.h"

namespace Torch {

	/** This class is designed to enhance an image using histogram equalisation

	    \begin{verbatim}

	         +---+          +--------------+         +---+
	         |xxx|	        |              |         |XXX|
		 |xxx|   ---->  | ipHistoEqual | ---->   |XXX|
	         |xxx|          |              |         |XXX|
	         +---+          +--------------+         +---+
	
	    \end{verbatim}
	    
	    @author Sebastien Marcel (marcel@idiap.ch)
	    @author Laurent El Shafey (laurent.el-shafey@idiap.ch)
	    @version 2.0
	    \date
	    @since 2.0
	*/

	class ipHistoEqual : public ipCore
	{
	public:

		/// constructor
	    	ipHistoEqual();

		/// destructor
		virtual ~ipHistoEqual();
	
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

