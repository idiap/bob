/**
 * @file cxx/old/ip/ip/ipBlock.h
 * @date Wed Apr 27 20:58:52 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
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
#ifndef _TORCHSPRO_IP_BLOCK_H_
#define _TORCHSPRO_IP_BLOCK_H_

#include "ip/ipCore.h"		

namespace Torch
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::ipCrop
	//	This class is designed to crop an image in multiple blocks.
	//	The result contains as many tensors of the same storage type
	//	than blocks.
	//
	//	- PARAMETERS (name, type, default value, description):
	//		"w"	int	0	"width of the block"
	//		"h"	int	0	"height of the block"
	//		"ox"	int	0	"number of overlapping pixels for blocks on the x axis"
	//		"oy"	int	0	"number of overlapping pixels for blocks on the y axis"
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class ipBlock : public ipCore
	{
	public:

		// Constructor
		ipBlock();

		// Destructor
		virtual ~ipBlock();

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

		int delta_block_overlap_x;
		int delta_block_overlap_y;
		int row_offset;
		int col_offset;
		int n_blocks_rows;
		int n_blocks_columns;
		int n_blocks;

		//
	};
}

#endif
