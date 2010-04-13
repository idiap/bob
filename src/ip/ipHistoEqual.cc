#include "ipHistoEqual.h"

/////////////////////////////////////////////////////////////////////////////////////////

namespace Torch {

/////////////////////////////////////////////////////////////////////////////////////////
// Constructor
ipHistoEqual::ipHistoEqual() 
	    : 	ipCore()
{
}

/////////////////////////////////////////////////////////////////////////////////////////
// Destructor
ipHistoEqual::~ipHistoEqual()
{
}

/////////////////////////////////////////////////////////////////////////////////////////
// Check if the input tensor has the right dimensions and type
bool ipHistoEqual::checkInput(const Tensor& input) const
{
	// Accept only 3D tensors of Torch::Image type
	if (    input.nDimension() != 3 ||
	        input.getDatatype() != Tensor::Short
		)
	{
		warning("ipHistoEqual::checkInput(): Incorrect Tensor type and dimension.");
		return false;
	}
	// Accept only grayscale images
	if (	input.size(2) !=1 )
	{
		warning("ipHistoEqual::checkInput(): Non grayscale image (multiple channels).");
		return false;
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////////////////////
// Allocate (if needed) the output tensors given the input tensor dimensions
bool ipHistoEqual::allocateOutput(const Tensor& input)
{
	if (	m_output == 0 ||
		m_output[0]->nDimension() != 3 ||
		m_output[0]->size(0) != input.size(0) ||
		m_output[0]->size(0) != input.size(1) ||
		m_output[0]->size(0) != input.size(2) )
	{
		cleanup();

		// Need allocation
		m_n_outputs = 1;
		m_output = new Tensor*[m_n_outputs];
		m_output[0] = new ShortTensor(input.size(0), input.size(1), input.size(2));
		return true;
	}

	return true;
}

/////////////////////////////////////////////////////////////////////////////////////////
// Process some input tensor (the input is checked, the outputs are allocated)
bool ipHistoEqual::processInput(const Tensor& input)
{
	// Process single channel images

	// Prepare pointers to access pixels
	const ShortTensor* t_input = (ShortTensor*)&input;
	ShortTensor* t_output = (ShortTensor*)m_output[0];

	const short* src = (const short*)t_input->dataR();
	short* dst = (short*)t_output->dataW();

        const int src_stride_h = t_input->t->stride[0];     // height
        const int src_stride_w = t_input->t->stride[1];     // width

        const int dst_stride_h = t_output->t->stride[0];     // height
        const int dst_stride_w = t_output->t->stride[1];     // width

        // An index for the 3D tensor is: [y * stride_h + x * stride_w + p * stride_p]
        const int height = input.size(0);
        const int width = input.size(1);
	const int wxh = width * height;

	// allocate temporary variables for histogram and cumulative histogram (1 plane)
	double histogram[256];
	double cumulHistogram[256];

	// Performs Histogram Equalization... (only one channel)

	// init histograms
	for(int i = 0; i < 256; i++)
	{
		histogram[i] = 0.0;
		cumulHistogram[i] = 0.0;
	}

	// compute histograms
	for ( int y = 0; y < height; y++ )
	{
		const short* src_row=&src[ y*src_stride_h ];
		for (int x = 0; x < width; x++, src_row+=src_stride_w )
		{
			short k = *src_row;
			if(k<0 || k>255) warning("ipHistoEqual::process() out-of-range value.");
			histogram[ k ]+=1.0 / wxh;
		}
	}
		
	// compute cumulative histogram
	double sum = 0.0;
        for (int i = 0; i < 256; i++ )
	{
		sum += histogram[ i ];
		cumulHistogram[ i ] = sum;
	}

	// normalize the histogram in the regular way
	for(int i = 0; i < 256; i++)
		cumulHistogram[i] = 255.0 * (cumulHistogram[i] - cumulHistogram[0]) / cumulHistogram[255];

	// fill in the output 
	for (int y = 0; y < height; y++ )
	{
		const short* src_row=&src[ y*src_stride_h ];
		short* dst_row=&dst[ y*dst_stride_h ];
		for (int x = 0; x < width; x++, src_row+=src_stride_w, dst_row+=dst_stride_w )
		{
			short k = *src_row; 
			// Convert the value into integer using the FixI macro
			*dst_row = FixI(cumulHistogram[k]);
		}
	}


	return true;
}

}

