#include "ipMSRSQIGaussian.h"

namespace Torch {

/////////////////////////////////////////////////////////////////////////
// Constructor

ipMSRSQIGaussian::ipMSRSQIGaussian()
	:	ipCore(),
		m_kernel(0)
{
	addIOption("RadiusX", 1, "Kernel radius on Ox");
	addIOption("RadiusY", 1, "Kernel radius on Oy");
	addDOption("Sigma", 5.0, "Variance of the kernel");
	addBOption("Weighed", false, "If true, SQI is performed (Weighed Gaussian kernel), otherwise MSR is done (Regular Gaussian kernel)");
}

/////////////////////////////////////////////////////////////////////////
// Destructor

ipMSRSQIGaussian::~ipMSRSQIGaussian()
{
	delete m_kernel;
}

//////////////////////////////////////////////////////////////////////////
// Check if the input tensor has the right dimensions and type

bool ipMSRSQIGaussian::checkInput(const Tensor& input) const
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

bool ipMSRSQIGaussian::allocateOutput(const Tensor& input)
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
// Allocate and compute the Gaussian kernel

void ipMSRSQIGaussian::prepareKernel(int radius_x, int radius_y, double sigma)
{
	// Check if the kernel needs allocation
	if (	m_kernel == 0 ||
		m_kernel->size(0) != (2 * radius_y + 1) ||
		m_kernel->size(1) != (2 * radius_x + 1))
	{
		delete m_kernel;
		m_kernel = new DoubleTensor(2 * radius_y + 1, 2 * radius_x + 1);
	}

	// Compute the kernel
	const double inv_sigma = 1.0  / sigma;
	double sum = 0.0;
  	for (int i = -radius_x; i <= radius_x; i ++)
    		for (int j = -radius_y; j <= radius_y; j ++)
    		{
      			const double weight = exp(- inv_sigma * (i * i + j * j));

			m_kernel->set(j + radius_y, i + radius_x, weight);
			sum += weight;
    		}

	// Normalize the kernel such that the sum over the area is equal to 1
  	const double inv_sum = 1.0 / sum ;
  	for (int i = -radius_x; i <= radius_x; i ++)
    		for (int j = -radius_y; j <= radius_y; j ++)
    		{
			m_kernel->set(	j + radius_y,
					i + radius_x,
					inv_sum * m_kernel->get(j + radius_y, i + radius_x));
		}
}

/////////////////////////////////////////////////////////////////////////
// Process some input tensor (the input is checked, the outputs are allocated)

bool ipMSRSQIGaussian::processInput(const Tensor& input)
{
	// Get the parameters
	const int radius_x = getIOption("RadiusX");
	const int radius_y = getIOption("RadiusY");
	const double sigma = getDOption("Sigma");
	const bool sqi = getBOption("Weighed");

	// Allocate and compute the kernel
	prepareKernel(radius_x, radius_y, sigma);

	// Prepare the input and output 3D image tensors
	const ShortTensor* t_input = (ShortTensor*)&input;
	ShortTensor* t_output = (ShortTensor*)m_output[0];

	const short* src = (const short*)t_input->dataR();
	short* dst = (short*)t_output->dataW();

	const int src_stride_h = t_input->t->stride[0];	// height
	const int src_stride_w = t_input->t->stride[1];	// width
	const int src_stride_p = t_input->t->stride[2];	// no planes

	const int dst_stride_h = t_output->t->stride[0];	// height
	const int dst_stride_w = t_output->t->stride[1];	// width
	const int dst_stride_p = t_output->t->stride[2];	// no planes

	// An index for the 3D tensor is: [y * stride_h + x * stride_w + p * stride_p]

	const int height = input.size(0);
	const int width = input.size(1);
	const int n_planes = input.size(2);

	const int start_x = radius_x;
	const int start_y = radius_y;
	const int stop_x = width - radius_x;
	const int stop_y = height - radius_y;

	// Fill with 0 the output image (to clear boundaries)
	t_output->fill(0);

	// Declare variable for new Weighed kernel
	DoubleTensor* m_kernel_weighed = 0;

	
	// Apply the kernel to the image for each color plane
	for (int p = 0; p < n_planes; p ++)
	{
		const short* src_plane = &src[p * src_stride_p];
		short* dst_plane = &dst[p * dst_stride_p];

		for (int y = 0; y < height; y ++)
		{
			short* dst_row = &dst_plane[ y * dst_stride_h ];
			for (int x = 0; x < width; x ++, dst_row += dst_stride_w)
			{
				// Weighed kernel for SQI
				if( sqi )
				{
					double region_mean;
					int under;
					int over;
					bool above;

					m_kernel_weighed = new DoubleTensor(2 * radius_y + 1, 2 * radius_x + 1);

					// prepare variables for an efficient access to the kernel values
					double* kernw = m_kernel_weighed->t->storage->data + m_kernel_weighed->t->storageOffset;
					const int kernw_stride_h = m_kernel_weighed->t->stride[0];	// height
					const int kernw_stride_w = m_kernel_weighed->t->stride[1];	// width


					// Init kernel weighed
					m_kernel_weighed->copy( m_kernel );
	
					// Compute region mean
					region_mean = 0.;
					for (int yy = -radius_y; yy <= radius_y; yy ++)
					{
						// Mirror interpolation
						int yyy=yy+y;
						if (yyy<0)
							yyy=abs(yyy)-1;
						if (yyy>=height)
							yyy=2*height-yyy-1;

						const short* src_row=&src[ yyy * src_stride_h ];	
						for (int xx = -radius_x; xx <= radius_x; xx ++)
						{
							// mirror interpolation
							int xxx=xx+x;
							if (xxx<0)
								xxx=abs(xxx)-1;
							if (xxx>=width)
								xxx=2*width-xxx-1;
					
							region_mean += src_row[ xxx * src_stride_w];
						}
					}
					region_mean /= ((2*radius_x+1)*(2*radius_y+1));
				
	
					// count number of pixels bigger/smaller than the mean
					under = 0;
					over = 0;	
					for (int yy = -radius_y; yy <= radius_y; yy ++)
					{
						// Mirror interpolation
						int yyy=yy+y;
						if (yyy<0)
							yyy=abs(yyy)-1;
						if (yyy>=height)
							yyy=2*height-yyy-1;

						const short* src_row=&src[ yyy * src_stride_h ];	
						for (int xx = -radius_x; xx <= radius_x; xx ++)
						{	
							// mirror interpolation
							int xxx=xx+x;
							if (xxx<0)
								xxx=abs(xxx)-1;
							if (xxx>=width)
								xxx=2*width-xxx-1;

							if(src_row[ xxx * src_stride_w]>region_mean)
								over++;
							else
								under++;
						}
					}
					if (over>under)
						above=true;
					else
						above=false;

				
					// Update filter weights 
					for (int yy = -radius_y; yy <= radius_y; yy ++)
					{
						// Mirror interpolation
						int yyy=yy+y;
						if (yyy<0)
							yyy=abs(yyy)-1;
						if (yyy>=height)
							yyy=2*height-yyy-1;

						const short* src_row=&src[ yyy * src_stride_h ];	
						double *kernw_row=&kernw[ (yy + radius_y) * kernw_stride_h ];
						for (int xx = -radius_x; xx <= radius_x; xx ++)
						{
							int xxx=xx+x;
							if (xxx<0)
								xxx=abs(xxx)-1;
							if (xxx>=width)
								xxx=2*width-xxx-1;

							if ( ((src_row[ xxx * src_stride_w] > region_mean) && !above)
							   || ((src_row[ xxx * src_stride_w] < region_mean) && above) )
							{	
								kernw_row[ (xx+radius_x) * kernw_stride_w ] = 0.;
							}
						}
					}

					// Normalize kernel
					double weighed_sum=0.;
					for (int yy = -radius_y; yy <= radius_y; yy ++)
					{
						double *kernw_row=&kernw[ (yy + radius_y) * kernw_stride_h ];
						for (int xx = -radius_x; xx <= radius_x; xx ++)
						{
							weighed_sum+=kernw_row[ (xx + radius_x) * kernw_stride_w ];
						}
					}
			
					for (int yy = -radius_y; yy <= radius_y; yy ++)
					{
						double *kernw_row=&kernw[ (yy + radius_y) * kernw_stride_h ];
						for (int xx = -radius_x; xx <= radius_x; xx ++)
						{
							kernw_row[ (xx + radius_x) * kernw_stride_w ] /= weighed_sum ;
						}
					}

					// Apply the kernel for the <y, x> pixel
					double sum = 0.0;
					for (int yy = -radius_y; yy <= radius_y; yy ++)
					{
						// mirror interpolation
						int yyy=yy+y;
						if (yyy<0)
							yyy=abs(yyy)-1;
						if (yyy>=height)
							yyy=2*height-yyy-1;
					
						const short* src_row=&src[ yyy * src_stride_h ];	
						double *kernw_row=&kernw[ (yy + radius_y) * kernw_stride_h ];
						for (int xx = -radius_x; xx <= radius_x; xx ++)
						{
							// mirror interpolation
							int xxx=xx+x;
							if (xxx<0)
								xxx=abs(xxx)-1;
							if (xxx>=width)
								xxx=2*width-xxx-1;

							sum +=  kernw_row[ (xx + radius_x) * kernw_stride_w ] * 
								src_row[ xxx * src_stride_w];
						}
					}
					delete m_kernel_weighed;
				
					// Update output using the FixI macro (round double value)
					*dst_row = FixI(sum);
				
				}
				// Regular Gaussian kernel for MSR
				else
				{	
					// prepare variables for an efficient access to the kernel values
					const double* kern = m_kernel->t->storage->data + m_kernel->t->storageOffset;
					const int kern_stride_h = m_kernel->t->stride[0];	// height
					const int kern_stride_w = m_kernel->t->stride[1];	// width

					// Apply the kernel for the <y, x> pixel
					double sum = 0.0;
					int yyy, xxx;
					for (int yy = -radius_y; yy <= radius_y; yy ++)
					{
						// mirror interpolation
						yyy=yy+y;
						if (yyy<0)
							yyy=abs(yyy)-1;
						if (yyy>=height)
							yyy=2*height-yyy-1;
				
						const short* src_row=&src_plane[ yyy * src_stride_h ];	
						const double *kern_row=&kern[ (yy + radius_y) * kern_stride_h ];
						for (int xx = -radius_x; xx <= radius_x; xx ++, kern_row+=kern_stride_w )
						{
							// mirror interpolation
							xxx=xx+x;
							if (xxx<0)
								xxx=abs(xxx)-1;
							if (xxx>=width)
								xxx=2*width-xxx-1;

							sum +=  *kern_row * src_row[ xxx * src_stride_w];
						}
					}
					// Update output using the FixI macro (round double value)
					*dst_row = FixI(sum);
		
				}

			}
		}
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////

}

