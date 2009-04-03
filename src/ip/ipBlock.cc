#include "ipBlock.h"
#include "Tensor.h"

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

	int n_blocks_columns = image_w / block_w;
	int n_blocks_rows = image_h / block_h;
	int n_blocks = n_blocks_columns * n_blocks_rows;

	print("ipBlock::allocateOutput()\n", n_blocks);
	print("   image width: %d\n", image_w);
	print("   image height: %d\n", image_h);
	print("   block width: %d\n", block_w);
	print("   block height: %d\n", block_h);
	print("   overlap x: %d\n", overlap_x);
	print("   overlap y: %d\n", overlap_y);
	print("\n");
	print("   number of blocks determined: %d\n", n_blocks);

	// Allocate output if required
	cleanup();

	// Need allocation

	/*
		optionally it might be better to output a 4D tensor: n_blocks_rows X n_blocks_cols X block_h X block_w
	*/

	m_n_outputs = n_blocks;
	m_output = new Tensor*[m_n_outputs];
	for(int i = 0 ; i < n_blocks ; i++)
		m_output[i] = new ShortTensor(block_h, block_w, input.size(2));
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Process some input tensor (the input is checked, the outputs are allocated)

bool ipBlock::processInput(const Tensor& input)
{
	// Get parameters
	const int overlap_x = getIOption("ox");
	const int overlap_y = getIOption("oy");
	const int block_w = getIOption("w");
	const int block_h = getIOption("h");

	// Prepare direct access to data
	const ShortTensor* t_input = (ShortTensor*)&input;
	ShortTensor* t_output = (ShortTensor*)m_output[0];

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////

}
