#include "ip/ipRescaleGray.h"
//STL #include <limits>

#define COMPUTE_SCALE_GRAY(tensorType, dataType)                                                                     \
{   \
  const tensorType* t_input = (const tensorType*)&input; \
  const int height = t_input->size(0);                                                                          \
  const int width = t_input->size(1);                                                                          \
  const int n_planes = t_input->size(2);                                                                          \
                                                                                                                \
  /* Start to "normalize" current values in range [0,255] */                                                   \
	double max_val = (*t_input)(0,0,0); /*STL std::numeric_limits<double>::min( ); */                                       \
	double min_val = (*t_input)(0,0,0); /*STL std::numeric_limits<double>::max( ); */                                       \
	double range;                                                                                                \
                                                                                                                     \
	/* find min and max values in the image */                                                                   \
	for( int p=0; p<n_planes; p++ )                                                                              \
	{                                                                                                            \
		for( int y=0; y<height; y++ )                                                                        \
		{                                                                                                    \
			for( int x=0; x<width; x++ )                                        \
			{                                                                                            \
				if ((*t_input)(y,x,p) > max_val)                                                              \
					max_val = (*t_input)(y,x,p);                                                          \
                                                                                                                     \
				if ((*t_input)(y,x,p) < min_val)                                                              \
					min_val = (*t_input)(y,x,p);                                                          \
			}                                                                                            \
		}                                                                                                    \
	}                                                                                                            \
                                                                                                                     \
	/* Compute the range */                                                                                      \
	range = max_val - min_val;                                                                                   \
	const double EPSILON = 1e-12; /*STL std::numeric_limits<double>::epsilon() * 1000; */                        \
	bool range_zero = range < EPSILON;                                                                           \
                                                                                                                     \
	/* Change the scale */                                                                                       \
	for( int p=0; p<n_planes; p++ )                                                                              \
	{                                                                                                            \
		for( int y=0; y<height; y++ )                                                                        \
		{                                                                                                    \
			for( int x=0; x<width; x++ )                \
			{                                                                                            \
				(*t_output)(y,x,p) = ( range_zero ? 0 : FixI(255. * ((*t_input)(y,x,p) - min_val) / range) );           \
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

    default:
      break;
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////

}

