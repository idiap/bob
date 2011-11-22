/**
 * @file cxx/ip/ip/ipTanTriggs.h
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
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

