#ifndef _IP_TAN_TRIGGS_H_
#define _IP_TAN_TRIGGS_H_

#include "ip/ipCore.h"

namespace Torch {

	/** This class is designed to perform the preprocessing chain of Tan and Triggs
	 *  to normalize images
	 *
	 *  @inproceedings{DBLP:conf/amfg/TanT07,
	 *    author    = {Xiaoyang Tan and
	 *                 Bill Triggs},
	 *    title     = {Enhanced Local Texture Feature Sets for Face Recognition
	 *                 Under Difficult Lighting Conditions},
	 *                 booktitle = {AMFG},
	 *    year      = {2007},
	 *    pages     = {168-182},
	 *    ee        = {http://dx.doi.org/10.1007/978-3-540-75690-3_13},
	 *    crossref  = {DBLP:conf/amfg/2007},
	 *    bibsource = {DBLP, http://dblp.uni-trier.de}
	 *  }
	 *

	    \verbatim

	         +---+          +---------------------+         +---+
	         |xxx|	        |                     |         |XXX|
		 |xxx|   ---->  |     ipTanTriggs     | ---->   |XXX|
	         |xxx|          |                     |         |XXX|
	         +---+          +---------------------+         +---+
	
	    \endverbatim

	    @author Laurent El Shafey (laurent.el-shafey@idiap.ch)
	    
	*/

	class ipTanTriggs : public ipCore
	{
	public:

		/// constructor
	    	ipTanTriggs();

		/// destructor
		virtual ~ipTanTriggs();
	
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

		// Compute the DoG kernel
		DoubleTensor*		computeDoG(double sigma0, double sigma1, int size);

	};

}

#endif

