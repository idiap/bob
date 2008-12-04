#include "ipLBP4R.h"
#include "Tensor.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
// Compute the 4R LBP code for a generic tensor

#define COMPUTE_LBP4R(tensorType, dataType)							\
{												\
	const tensorType* t_input = (tensorType*)&input;					\
												\
	const dataType* src = t_input->t->storage->data + t_input->t->storageOffset;		\
												\
	const int in_stride_h = t_input->t->stride[0];						\
	const int in_stride_w = t_input->t->stride[1];						\
	const int in_stride_p = t_input->t->stride[2];						\
												\
	dataType tab[4];									\
	tab[0] = src[(m_y - m_R) * in_stride_h + m_x * in_stride_w];				\
	tab[1] = src[m_y * in_stride_h + (m_x + m_R) * in_stride_w];				\
	tab[2] = src[(m_y + m_R) * in_stride_h + m_x * in_stride_w];				\
	tab[3] = src[m_y * in_stride_h + (m_x - m_R) * in_stride_w];				\
												\
	const dataType center = src[m_y * in_stride_h + m_x * in_stride_w];			\
												\
	const dataType cmp_point = m_toAverage ?						\
		(dataType)									\
                        (0.5 + 0.2 * (tab[0] + tab[1] + tab[2] + tab[3] + center + 0.0))	\
		:										\
		center;										\
												\
	unsigned char lbp = 0;									\
												\
	lbp = lbp << 1;										\
	if (tab[0] > cmp_point) lbp ++;								\
	lbp = lbp << 1;										\
	if (tab[1] > cmp_point) lbp ++;								\
	lbp = lbp << 1;										\
	if (tab[2] > cmp_point) lbp ++;								\
	lbp = lbp << 1;										\
	if (tab[3] > cmp_point) lbp ++;								\
	if (m_addAvgBit == true && m_rot_invariant == false && m_uniform == false)              \
	{                                                                                       \
		lbp = lbp << 1;                                                                 \
		if (center > cmp_point) lbp ++;                                                 \
	}                                                                                       \
                                                                                                \
	*m_lbp = m_rot_invariant ?                                                              \
			(m_uniform ? m_lut_U2RI[lbp] : m_lut_RI[lbp]) :                         \
			(m_uniform ? m_lut_U2[lbp] : lbp);                                      \
}

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace Torch {

///////////////////////////////////////////////////////////////////////////////////////////////////
// Constructor

ipLBP4R::ipLBP4R(int R)
	:	ipLBP(4, R),
                m_lut_RI (0),
		m_lut_U2 (0),
		m_lut_U2RI (0)
{
        m_lut_RI = new unsigned char [16];
	m_lut_U2 = new unsigned char [16];
	m_lut_U2RI = new unsigned char [16];
	for (int i=0; i<16; i++)
	{
		m_lut_RI[i] = 0;
		m_lut_U2[i] = 0;
		m_lut_U2RI[i] = 0;
	}
	init_lut_RI();
	init_lut_U2();
	init_lut_U2RI();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Destructor

ipLBP4R::~ipLBP4R()
{
        delete[] m_lut_RI;
	delete[] m_lut_U2;
	delete[] m_lut_U2RI;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Get the maximum possible label

int ipLBP4R::getMaxLabel()
{
	return	m_rot_invariant ?
			(m_uniform ? 	6 	// Rotation invariant + uniform
					:
					6)	// Rotation invariant
			:
			(m_uniform ?	15	// Uniform
					:
					(m_toAverage ?
						(m_addAvgBit ? 32 : 16)	// i.e. 2^5=32 vs. 2^4=16
						:
						16)				// i.e. 2^4=16)
			);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Process some input tensor (the input is checked, the outputs are allocated) - overriden

bool ipLBP4R::processInput(const Tensor& input)
{
	switch (input.getDatatype())
	{
	case Tensor::Char:
		COMPUTE_LBP4R(CharTensor, char);
		break;

	case Tensor::Short:
		COMPUTE_LBP4R(ShortTensor, short);
		break;

	case Tensor::Int:
		COMPUTE_LBP4R(IntTensor, int);
		break;

	case Tensor::Long:
		COMPUTE_LBP4R(LongTensor, long);
		break;

	case Tensor::Float:
		COMPUTE_LBP4R(FloatTensor, float);
		break;

	case Tensor::Double:
		COMPUTE_LBP4R(DoubleTensor, double);
		break;
	}

	return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void ipLBP4R::init_lut_RI()
{
	// all 0's
	m_lut_RI[0] = 1;
	// 3 0's + 1 1's
	m_lut_RI[1] = 2;
	m_lut_RI[2] = 2;
	m_lut_RI[4] = 2;
	m_lut_RI[8] = 2;
	// 2 0's + 2 1's
	m_lut_RI[3] = 3;
	m_lut_RI[5] = 3;
	m_lut_RI[6] = 3;
	m_lut_RI[9] = 3;
	m_lut_RI[10] = 3;
	m_lut_RI[12] = 3;
	// 1 0's + 3 1's
	m_lut_RI[7] = 4;
	m_lut_RI[11] = 4;
	m_lut_RI[13] = 4;
	m_lut_RI[14] = 4;
	// all 1's
	m_lut_RI[15] = 5;

}

///////////////////////////////////////////////////////////////////////////////////////////////////

void ipLBP4R::init_lut_U2()
{
	// A) all non uniform patterns have a label of 0.
	// already initialized to 0

	// B) LBP pattern with 0 bit to 1
	m_lut_U2[0] = 1;

	// C) LBP patterns with 1 bit to 1
	m_lut_U2[8] = 2;
	m_lut_U2[4] = 3;
	m_lut_U2[2] = 4;
	m_lut_U2[1] = 5;

	// D) LBP patterns with 2 bits to 1
	m_lut_U2[8+4] = 6;
	m_lut_U2[4+2] = 7;
	m_lut_U2[2+1] = 8;
	m_lut_U2[1+8] = 9;

	// E) LBP patterns with 3 bits to 1
	m_lut_U2[8+4+2] = 10;
	m_lut_U2[4+2+1] = 11;
	m_lut_U2[2+1+8] = 12;
	m_lut_U2[1+8+4] = 13;

	// F) LBP patterns with 4 bits to 1
	m_lut_U2[8+4+2+1] = 14;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void ipLBP4R::init_lut_U2RI()
{
	// A) all non uniform patterns have a label of 0.
	// already initialized to 0

	// All bits are 0
	m_lut_U2RI[0] = 1;

	// only one bit is 1 rest are 0's
	m_lut_U2RI[1] = 2;
	m_lut_U2RI[2] = 2;
	m_lut_U2RI[4] = 2;
	m_lut_U2RI[8] = 2;

	// only  two adjacent bits are 1 rest are 0's
	m_lut_U2RI[3] = 3;
	m_lut_U2RI[6] = 3;
	m_lut_U2RI[12] = 3;

	// only three adjacent bits are 1 rest are 0's
	m_lut_U2RI[7] = 4;
	m_lut_U2RI[14] = 4;

	// four adjacent bits are 1
	m_lut_U2RI[15] = 5;

}

///////////////////////////////////////////////////////////////////////////////////////////////////

}
