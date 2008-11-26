#include "ipLBP4R.h"
#include "Tensor.h"

namespace Torch {

///////////////////////////////////////////////////////////////////////////////////////////////////
// Constructor

ipLBP4R::ipLBP4R(int R)
	:	ipLBP(4, R)
{
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Destructor

ipLBP4R::~ipLBP4R()
{
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Get the maximum possible label

int ipLBP4R::getMaxLabel()
{
	return m_toAverage ?
		(m_addAvgBit ? 32 : 16)	// i.e. 2^5=32 vs. 2^4=16
		:
		16;			// i.e. 2^4=16
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Process some input tensor (the input is checked, the outputs are allocated) - overriden

bool ipLBP4R::processInput(const Tensor& input)
{
	// Prepare the input arrays
	const ShortTensor* t_input = (ShortTensor*)&input;

	const short* src = t_input->t->storage->data + t_input->t->storageOffset;

	const int in_stride_h = t_input->t->stride[0];	// height
	const int in_stride_w = t_input->t->stride[1];	// width
	const int in_stride_p = t_input->t->stride[2];	// no planes

	// Prepare the data for LBP computation (only the first color channel is used)
	int tab[4];
	tab[0] = src[(m_y - m_R) * in_stride_h + m_x * in_stride_w];
	tab[1] = src[m_y * in_stride_h + (m_x + m_R) * in_stride_w];
	tab[2] = src[(m_y + m_R) * in_stride_h + m_x * in_stride_w];
	tab[3] = src[m_y * in_stride_h + (m_x - m_R) * in_stride_w];

	const int center = src[m_y * in_stride_h + m_x * in_stride_w];

	const int cmp_point = m_toAverage ?
		(int)
                        (0.5 + 0.2 * (tab[0] + tab[1] + tab[2] + tab[3] + center + 0.0))
		:
		center;

	// Compute the LBP (a byte is enough as 5 bits maximum are required)
	unsigned char lbp = 0;

	lbp = lbp << 1;
	if (tab[0] > cmp_point) lbp ++;
	lbp = lbp << 1;
	if (tab[1] > cmp_point) lbp ++;
	lbp = lbp << 1;
	if (tab[2] > cmp_point) lbp ++;
	lbp = lbp << 1;
	if (tab[3] > cmp_point) lbp ++;
	if (m_addAvgBit)
	{
		lbp = lbp << 1;
		if (center > cmp_point) lbp ++;
	}

	// OK (direct access the output tensor)
	*m_lbp = lbp;
	return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

}
