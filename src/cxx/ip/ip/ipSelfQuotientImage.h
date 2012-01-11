/**
 * @file cxx/ip/ip/ipSelfQuotientImage.h
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
#ifndef _BOB5SPRO_IP_SELF_QUOTIENT_IMAGE_H_
#define _BOB5SPRO_IP_SELF_QUOTIENT_IMAGE_H_

#include "ip/ipCore.h"

namespace bob {

	/** This class is designed to perform the Self Quotient Image algorithm on an image

	    \verbatim

	         +---+          +---------------------+         +---+
	         |xxx|	        |                     |         |XXX|
		 |xxx|   ---->  | ipSelfQuotientImage | ---->   |XXX|
	         |xxx|          |                     |         |XXX|
	         +---+          +---------------------+         +---+
	
	    \endverbatim

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

