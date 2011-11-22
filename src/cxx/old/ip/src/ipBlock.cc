/**
 * @file cxx/old/ip/src/ipBlock.cc
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
#include "ip/ipBlock.h"
#include "core/Tensor.h"

namespace Torch {

/////////////////////////////////////////////////////////////////////////
// Constructor

ipBlock::ipBlock()
	:	ipCore()
{
	addIOption("w", 0, "width of the block");
	addIOption("h", 0, "height of the block");
	addIOption("ox", 0, "number of overlapping pixels for blocks on the x axis");
	addIOption("oy", 0, "number of overlapping pixels for blocks on the y axis");
	addBOption("rcoutput", false, "creates a rows X columns output (for grayscale input only) and thus a single 4D output tensor");
	addBOption("verbose", false, "verbose");
}

/////////////////////////////////////////////////////////////////////////
// Destructor

ipBlock::~ipBlock()
{
}

//////////////////////////////////////////////////////////////////////////
// Check if the input tensor has the right dimensions and type

bool ipBlock::checkInput(const Tensor& input) const
{
	// Accept only 3D tensors of Torch::Image type
	if (	input.nDimension() != 3 ||
		input.getDatatype() != Tensor::Short)
	{
		return false;
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Allocate (if needed) the output tensors given the input tensor dimensions

bool ipBlock::allocateOutput(const Tensor& input)
{
	const int overlap_x = getIOption("ox");
	const int overlap_y = getIOption("oy");
	const int block_w = getIOption("w");
	const int block_h = getIOption("h");
	const bool verbose = getBOption("verbose");

	// Check parameters
	if (	block_w < 0 || block_h < 0 ||
		block_w > input.size(1) ||
		block_h > input.size(0))
	{
		return false;
	}

	if((overlap_x < 0) || (overlap_x > (block_w-1)))
	{
		warning("ipBlock: overlap x should be in range [0...%d].", block_w-1);
		return false;
	}

	if((overlap_y < 0) || (overlap_y > (block_h-1)))
	{
		warning("ipBlock: overlap y should be in range [0...%d].", block_h-1);
		return false;
	}

	int image_w = input.size(1);
	int image_h = input.size(0);

	delta_block_overlap_x = block_w - overlap_x;
	delta_block_overlap_y = block_h - overlap_y;
	
	n_blocks_rows = 0;
	n_blocks_columns = 0;
	int r, c;

	for(r = 1 ; r <= image_h - (block_h - 1) ; r += delta_block_overlap_y) n_blocks_rows++;
	for(c = 1 ; c <= image_w - (block_w - 1) ; c += delta_block_overlap_x) n_blocks_columns++;
	
	int extra_rows = image_h - (r - delta_block_overlap_y + (block_h-1));
	int extra_cols = image_w - (c - delta_block_overlap_x + (block_w-1));
	
	row_offset = extra_rows/2;
	col_offset = extra_cols/2;

	n_blocks = n_blocks_columns * n_blocks_rows;

	if(verbose)
	{
		print("ipBlock::allocateOutput()\n", n_blocks);
		print("   image width: %d\n", image_w);
		print("   image height: %d\n", image_h);
		print("   block width: %d\n", block_w);
		print("   block height: %d\n", block_h);
		print("   overlap x: %d\n", overlap_x);
		print("   overlap y: %d\n", overlap_y);
		print("\n");
		print("   number of blocks determined: %d\n", n_blocks);
		print("   number of row blocks: %d\n", n_blocks_rows);
		print("   number of column blocks: %d\n", n_blocks_columns);
	}

	// Allocate output if required
	cleanup();

	// Need allocation

	/*
		optionally it might be better to output a 4D tensor: n_blocks_rows X n_blocks_cols X block_h X block_w
	*/
	const bool rcoutput = getBOption("rcoutput");
	if(rcoutput)
	{
		if(input.size(2) != 1)
		{
			warning("Impossible to create row X columns output for color images."); 
			return false;
		}

		if(verbose) message("Building row X columns output."); 

		m_n_outputs = 1;
		m_output = new Tensor*[1];
		m_output[0] = new ShortTensor(n_blocks_rows, n_blocks_columns, block_h, block_w);
	}
	else
	{
		m_n_outputs = n_blocks;
		m_output = new Tensor*[m_n_outputs];
		for(int i = 0 ; i < n_blocks ; i++)
			m_output[i] = new ShortTensor(block_h, block_w, input.size(2));
	}

	return true;
}

/////////////////////////////////////////////////////////////////////////
// Process some input tensor (the input is checked, the outputs are allocated)

bool ipBlock::processInput(const Tensor& input)
{
	// Get parameters
	const int block_w = getIOption("w");
	const int block_h = getIOption("h");
	const bool rcoutput = getBOption("rcoutput");

	// Prepare direct access to data
	const ShortTensor* t_input = (ShortTensor*)&input;
				
  	ShortTensor *t_input_narrow_rows = new ShortTensor();
  	ShortTensor *t_input_narrow_cols = new ShortTensor();

	ShortTensor* t_rcoutput = (ShortTensor*)m_output[0];
  	ShortTensor *t_rcoutput_narrow_rows = NULL;
  	ShortTensor *t_rcoutput_narrow_cols = NULL;
  	ShortTensor *t_ = NULL;

	//t_input->print("input");
	
	if(rcoutput)
	{
  		t_rcoutput_narrow_rows = new ShortTensor();
  		t_rcoutput_narrow_cols = new ShortTensor();
  		t_ = new ShortTensor();
	}

	for(int r = 0; r < n_blocks_rows; r++)
	{
		int row = row_offset + r * delta_block_overlap_y;

		// narrow the tensor t_input along rows (dimension 0) at row #row# and length #block_h#
		t_input_narrow_rows->narrow(t_input, 0, row, block_h);

		if(rcoutput)
		{
			// narrow the tensor t_rcoutput along block rows (dimension 0) at row #r# and length 1
			//t_rcoutput_narrow_rows->narrow(t_rcoutput, 0, r, 1);

		   	// but better to use a select to return a sub-tensor
			t_rcoutput_narrow_rows->select(t_rcoutput, 0, r);
		}

	   	for(int c = 0; c < n_blocks_columns; c++) 
		{
			int col = col_offset + c * delta_block_overlap_x;

			// narrow the tensor t_input_narrow_rows along columns (dimension 1) at column #col# and length #block_w#
			t_input_narrow_cols->narrow(t_input_narrow_rows, 1, col, block_w);

			if(rcoutput)
			{
				/*
					Warning: t_rcoutput_narrow_cols is a 4D tensor and t_input_narrow_cols a 3D tensor

					However as the lenght of narrow along each dim is 1 we don't need to do any selects
				*/

				// narrow the tensor t_rcoutput along block rows (dimension 0) at row #r# and length 1
				//t_rcoutput_narrow_cols->narrow(t_rcoutput_narrow_rows, 1, c, 1);
		   		
				// but better to use a select to return a sub-tensor
				t_rcoutput_narrow_cols->select(t_rcoutput_narrow_rows, 0, c); // here we should refer to 0 as we are using a 3D tensor now 
	
				//print("select col: %dD -> %dD\n", t_rcoutput_narrow_rows->nDimension(), t_rcoutput_narrow_cols->nDimension());


				// copy the block #t_input_narrow_cols# (3D tensor) into t_rcoutput_narrow_cols (4D tensor) 
				//t_rcoutput_narrow_cols->copy(t_input_narrow_cols);

				// but better to use a select to return a sub-tensor and copy it
				t_->select(t_input_narrow_cols, 2, 0);
				t_rcoutput_narrow_cols->copy(t_);

				//t_->print();
			}
			else
			{
				// Storing blocks as [colblock][rowblock] (i.e. [x][y])
		   		int index_block = c + r * n_blocks_columns;

				ShortTensor* t_output = (ShortTensor*)m_output[index_block];
		
				// copy he block #t_input_narrow_cols# (3D tensor) into the current output tensor (3D as well)
				t_output->copy(t_input_narrow_cols);
			}
		}
	}

	if(rcoutput)
	{
		delete t_;
		delete t_rcoutput_narrow_cols;
		delete t_rcoutput_narrow_rows;
	}
	delete t_input_narrow_cols;
	delete t_input_narrow_rows;

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////

}
