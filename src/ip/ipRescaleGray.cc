#include "ipRescaleGray.h"
#include <limits>

#define COMPUTE_SCALE_GRAY(tensorType, dataType)                                                                     \
{                                                                                                                    \
	const tensorType* t_input = (tensorType*)&input;                                                             \
	const dataType* src = (const dataType*)input.dataR();                                                        \
                                                                                                                     \
        const int src_stride_h = t_input->t->stride[0];     /* height */                                             \
        const int src_stride_w = t_input->t->stride[1];     /* width */                                              \
        const int src_stride_p = t_input->t->stride[2];     /* nb planes */                                          \
                                                                                                                     \
        /* An index for the 3D tensor is: [y * stride_h + x * stride_w + p * stride_p] */                            \
        const int height = input.size(0);                                                                            \
        const int width = input.size(1);                                                                             \
        const int n_planes = input.size(2);                                                                          \
                                                                                                                     \
        /* Start to "normalize" current values in range [0,255] */                                                   \
	double max_val = std::numeric_limits<double>::min( );                                                        \
	double min_val = std::numeric_limits<double>::max( );                                                        \
	double range;                                                                                                \
                                                                                                                     \
	/* find min and max values in the image */                                                                   \
	for( int p=0; p<n_planes; p++ )                                                                              \
	{                                                                                                            \
		const dataType* src_plane = &src[ p * src_stride_p ];                                                \
		for( int y=0; y<height; y++ )                                                                        \
		{                                                                                                    \
			const dataType* src_row = &src_plane[ y * src_stride_h ];                                    \
			for( int x=0; x<width; x++, src_row += src_stride_w )                                        \
			{                                                                                            \
				if (*src_row > max_val)                                                              \
					max_val = *src_row;                                                          \
                                                                                                                     \
				if (*src_row < min_val)                                                              \
					min_val = *src_row;                                                          \
			}                                                                                            \
		}                                                                                                    \
	}                                                                                                            \
                                                                                                                     \
	/* Compute the range */                                                                                      \
	range = max_val - min_val;                                                                                   \
	const double EPSILON = std::numeric_limits<double>::epsilon() * 1000;                                        \
	bool range_zero = range < EPSILON;                                                                           \
                                                                                                                     \
	/* Change the scale */                                                                                       \
	for( int p=0; p<n_planes; p++ )                                                                              \
	{                                                                                                            \
		const dataType* src_plane = &src[ p * src_stride_p ];                                                \
		short* dst_plane = &dst[ p * dst_stride_p ];                                                         \
		for( int y=0; y<height; y++ )                                                                        \
		{                                                                                                    \
			const dataType* src_row = &src_plane[ y * src_stride_h ];                                    \
			short* dst_row = &dst_plane[ y * dst_stride_h ];                                             \
			for( int x=0; x<width; x++, src_row += dst_stride_w, dst_row+= dst_stride_w )                \
			{                                                                                            \
				*dst_row = ( range_zero ? 0 : FixI(255. * (*src_row - min_val) / range) );           \
			}                                                                                            \
		}                                                                                                    \
	}                                                                                                            \
}                                                                                                                    

namespace Torch {

/////////////////////////////////////////////////////////////////////////
// Constructor

ipRescaleGray::ipRescaleGray()
	:	ipCore()
{
}

/////////////////////////////////////////////////////////////////////////
// Destructor

ipRescaleGray::~ipRescaleGray()
{
}

//////////////////////////////////////////////////////////////////////////
// Check if the input tensor has the right dimensions and type

bool ipRescaleGray::checkInput(const Tensor& input) const
{
	// Accept only 3D tensors 
	if (	input.nDimension() != 3 )
// || input.getDatatype() != Tensor::Short)
	{
		return false;
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Allocate (if needed) the output tensors given the input tensor dimensions

bool ipRescaleGray::allocateOutput(const Tensor& input)
{
	// Allocate output if required
	if (	m_output == 0 ||
		m_output[0]->nDimension() != 3 ||
		m_output[0]->size(0) != input.size(0) ||
		m_output[0]->size(1) != input.size(1) ||
		m_output[0]->size(2) != input.size(2))
	{
		cleanup();

		// Need allocation
		m_n_outputs = 1;
		m_output = new Tensor*[m_n_outputs];
		m_output[0] = new ShortTensor( input.size(0), input.size(1), input.size(2) );
		return true;
	}

	return true;
}

/////////////////////////////////////////////////////////////////////////
// Process some input tensor (the input is checked, the outputs are allocated)

bool ipRescaleGray::processInput(const Tensor& input)
{
	// Prepare direct access to output data
	ShortTensor* t_output = (ShortTensor*)m_output[0];	
	short* dst = (short*)t_output->dataW();                                                                   
                                              
        const int dst_stride_h = t_output->t->stride[0];     /* height */                                                 
        const int dst_stride_w = t_output->t->stride[1];     /* width */
        const int dst_stride_p = t_output->t->stride[2];     /* nb_planes */

	switch (input.getDatatype())
	{
		case Tensor::Char:
			COMPUTE_SCALE_GRAY(CharTensor, char);
			break;

		case Tensor::Short:
			COMPUTE_SCALE_GRAY(ShortTensor, short);
			break;

		case Tensor::Int:
			COMPUTE_SCALE_GRAY(IntTensor, int);
			break;

		case Tensor::Long:
			COMPUTE_SCALE_GRAY(LongTensor, long);
			break;

		case Tensor::Float:
			COMPUTE_SCALE_GRAY(FloatTensor, float);
			break;

		case Tensor::Double:
			COMPUTE_SCALE_GRAY(DoubleTensor, double);
			break;
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////

}

