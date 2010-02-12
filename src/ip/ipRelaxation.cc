#include "ipRelaxation.h"
#include "multigrid.h"
#include "ipRescaleGray.h"
#include <limits>

namespace Torch {

////////////////////////////////////////////////////////////////////
// Constructor
ipRelaxation::ipRelaxation() : ipCore()
{
	addIOption("type",1,"Type of diffusion (isotropic, anisotropic)");
	addIOption("steps",10,"Number of relaxation steps to approximate the solution");
	addDOption("lambda",5.,"Relative importance of the smoothness constraint");
}


////////////////////////////////////////////////////////////////////
// Destructor
ipRelaxation::~ipRelaxation() 
{
}


//////////////////////////////////////////////////////////////////////////
// Check if the input tensor has the right dimensions and type
bool ipRelaxation::checkInput(const Tensor& input) const
{
	// Accept only 3D tensors of Torch::Image type
	if (	input.nDimension() != 3 ||
		input.getDatatype() != Tensor::Short)
	{
		return false;
	}
	// Accept only gray images
	if (	input.size(2) !=1 )
	{
		warning("ipRelaxation::checkInput(): Non gray level image (multiple channels).");
		return false;
	}

	// OK
	return true;
}


/////////////////////////////////////////////////////////////////////////
// Allocate (if needed) the output tensors given the input tensor dimensions
bool ipRelaxation::allocateOutput(const Tensor& input)
{
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
		m_output[0] = new ShortTensor(input.size(0), input.size(1), input.size(2));
		return true;
	}

	return true;
}


/////////////////////////////////////////////////////////////////////////
// Process some input tensor (the input is checked, the outputs are allocated)
bool ipRelaxation::processInput(const Tensor& input)
{
	// Get the parameters
	const double lambda = getDOption("lambda");
	const int steps = getIOption("steps");
	const int type = getIOption("type");

	// Prepare the input and output 3D image tensors
	const ShortTensor* t_input = (ShortTensor*)&input;
	ShortTensor* t_output = (ShortTensor*)m_output[0];

	const short* src = t_input->t->storage->data + t_input->t->storageOffset;
	short* dst = t_output->t->storage->data + t_output->t->storageOffset;

	const int src_stride_h = t_input->t->stride[0];	// height
	const int src_stride_w = t_input->t->stride[1];	// width

	const int dst_stride_h = t_output->t->stride[0];	// height
	const int dst_stride_w = t_output->t->stride[1];	// width

	// An index for the 3D tensor is: [y * stride_h + x * stride_w + p * stride_p]

	const int height = input.size(0);
	const int width = input.size(1);

	DoubleTensor* rho = new DoubleTensor(5);
	DoubleTensor* image = new DoubleTensor(height,width,1);
	DoubleTensor* light = new DoubleTensor(height,width,1);

	// Initializes variables for efficient access
	double* img = image->t->storage->data + image->t->storageOffset;	
	double* lig = light->t->storage->data + light->t->storageOffset;	

	const int d_stride_h = image->t->stride[0];	// height
	const int d_stride_w = image->t->stride[1];	// width

	// initialization: "source" term: image | solution: light
	for (int y=0; y<height; y++)
	{	
		const short* src_row = &src[ y * src_stride_h ];
		int ind_h = y * d_stride_h;
		double* image_row = &img[ ind_h ];
		double* light_row = &lig[ ind_h ];
		for (int x=0; x<width; x++, image_row+= d_stride_w, light_row+=d_stride_w, src_row+=src_stride_w )
		{
			*image_row = *src_row;	
			*light_row = *src_row;	
		}
	}

	// apply relaxation steps (gaussSeidel -> see multigrid.cc)
	for (int i=0; i<=steps; i++)
	{ 
		gaussSeidel(*light, *image, *rho, lambda, type );

		// swap: the improved estimate becomes the new estimate for next relaxation step
		image->copy(light);
	}

	// Rescale the values in [0,255] and copy it into the output Tensor
	ipCore *rescale = new ipRescaleGray();
	CHECK_FATAL(rescale->process(*light) == true);
	light->copy( &(rescale->getOutput(0)) );
	delete rescale;
	
	// build final result (R = I/L)
	for(int y = 0 ; y < height ; y++)
	{
		const short* src_row = &src[ y * src_stride_h ];
		double* light_row = &lig[ y * d_stride_h ];
		for(int x = 0 ; x < width ; x++, src_row+=src_stride_w, light_row+=d_stride_w )
		{
			// Set R=I/L equal to 1 at the border
			if ((y == 0) || (y == height - 1) ||  (x == 0) || (x == width-1)) 
				*light_row = 1.;
			else 
			{
				if (IS_NEAR(*light_row, 0.0, 1)) 
					*light_row = 1.;  
				else
					*light_row = *src_row / *light_row;
			}
		}
	}

       	cutExtremum(*light, 4); 
       
 
	// Rescale the values in [0,255] and copy it into the output Tensor
	rescale = new ipRescaleGray();
	CHECK_FATAL(rescale->process(*light) == true);
	t_output->copy( &(rescale->getOutput(0)) );

	// Clean up
	delete rescale;
  
	delete rho;
	delete image;
	delete light;

	return true;
}


bool ipRelaxation::cutExtremum(DoubleTensor& data, int distribution_width) 
{
	DoubleTensor* t_data = (DoubleTensor*)&data;	
	double* dat = t_data->t->storage->data + t_data->t->storageOffset;
 
	const int stride_h = t_data->t->stride[0];	// height
	const int stride_w = t_data->t->stride[1];	// width

	// An index for the 3D tensor is: [y * stride_h + x * stride_w + p * stride_p]
	const int height = data.size(0);
	const int width = data.size(1);
	const int wxh = width * height;

	// used to 'cut' the extreme of the pixels distribution in the result
	double mean_out = 0.0;
	double var_out = 0.0;
	double std_dev = 0.0; 
    
	// compute the mean
	for(int y = 0 ; y < height ; y++)
	{
		double* t_data_row = &dat[ y * stride_h ];
		for(int x = 0 ; x < width ; x++, t_data_row+=stride_w )
		{
			mean_out += *t_data_row;
		}
	}
	mean_out /= wxh;
    
	// compute variance and standard deviation
	for(int y = 0 ; y < height ; y++)
	{
		double* t_data_row = &dat[ y * stride_h ];
		for(int x = 0 ; x < width ; x++, t_data_row+=stride_w )
		{
			var_out += ( *t_data_row - mean_out ) * ( *t_data_row - mean_out );    
		}
	}
	var_out /= (wxh - 1);
	std_dev = sqrt(var_out);

	/// Cut
	double mean_plus_dxstd = mean_out + distribution_width*std_dev;
	double mean_minus_dxstd = mean_out - distribution_width*std_dev;
	
	for(int y = 0 ; y < height ; y++)
	{
		double* t_data_row = &dat[ y * stride_h ];
		for(int x = 0 ; x < width ; x++, t_data_row+=stride_w )
		{
			if ( *t_data_row > mean_plus_dxstd )
				*t_data_row = mean_plus_dxstd;
      
			if ( *t_data_row < mean_minus_dxstd )
				*t_data_row = mean_minus_dxstd;
		}
	}

	return true;
}


}
