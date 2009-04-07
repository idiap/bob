#ifndef _TORCHSPRO_IP_BLOCK_H_
#define _TORCHSPRO_IP_BLOCK_H_

#include "ipCore.h"		

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
