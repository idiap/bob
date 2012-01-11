/**
 * @file cxx/ip/src/ipMSRSQIGaussian.cc
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
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
#include "ip/ipMSRSQIGaussian.h"

namespace bob {

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
	// Accept only 3D tensors of bob::Image type
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
      (*m_kernel)(j + radius_y, i + radius_x) = weight;
			sum += weight;
    }

	// Normalize the kernel such that the sum over the area is equal to 1
 	const double inv_sum = 1.0 / sum ;
 	for (int i = -radius_x; i <= radius_x; i ++)
 		for (int j = -radius_y; j <= radius_y; j ++)
 		{
			(*m_kernel)(	j + radius_y,	i + radius_x) *= inv_sum;
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

	const int src_stride_h = t_input->stride(0);	// height
	const int src_stride_w = t_input->stride(1);	// width
	const int src_stride_p = t_input->stride(2);	// no planes


	// An index for the 3D tensor is: [y * stride_h + x * stride_w + p * stride_p]

	const int height = input.size(0);
	const int width = input.size(1);
	const int n_planes = input.size(2);

	// Fill with 0 the output image (to clear boundaries)
	t_output->fill(0);

	// Declare variable for new Weighed kernel
	DoubleTensor* m_kernel_weighed = 0;

	
	// Apply the kernel to the image for each color plane
	for (int p = 0; p < n_planes; p++)
	{
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				// Weighed kernel for SQI
				if( sqi )
				{
					double region_mean;
					int under;
					int over;
					bool above;

					m_kernel_weighed = new DoubleTensor(2 * radius_y + 1, 2 * radius_x + 1);
					// Init kernel weighed
					m_kernel_weighed->copy( m_kernel );

					// Compute region mean
					region_mean = 0.;
					for (int yy = -radius_y; yy <= radius_y; yy++)
					{
						// Mirror interpolation
						int yyy=yy+y;
						if (yyy<0)
							yyy=abs(yyy)-1;
						if (yyy>=height)
							yyy=2*height-yyy-1;

						for (int xx = -radius_x; xx <= radius_x; xx++)
						{
							// mirror interpolation
							int xxx=xx+x;
							if (xxx<0)
								xxx=abs(xxx)-1;
							if (xxx>=width)
								xxx=2*width-xxx-1;
					
							region_mean += (*t_input)(yyy * src_stride_h + xxx * src_stride_w + p * src_stride_p );
						}
					}
					region_mean /= ((2*radius_x+1)*(2*radius_y+1));
				
	
					// count number of pixels bigger/smaller than the mean
					under = 0;
					over = 0;	
					for (int yy = -radius_y; yy <= radius_y; yy++)
					{
						// Mirror interpolation
						int yyy=yy+y;
						if (yyy<0)
							yyy=abs(yyy)-1;
						if (yyy>=height)
							yyy=2*height-yyy-1;

						for (int xx = -radius_x; xx <= radius_x; xx++)
						{	
							// mirror interpolation
							int xxx=xx+x;
							if (xxx<0)
								xxx=abs(xxx)-1;
							if (xxx>=width)
								xxx=2*width-xxx-1;

							if((*t_input)( yyy* src_stride_h + xxx * src_stride_w + p * src_stride_p)>region_mean)
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
					for (int yy = -radius_y; yy <= radius_y; yy++)
					{
						// Mirror interpolation
						int yyy=yy+y;
						if (yyy<0)
							yyy=abs(yyy)-1;
						if (yyy>=height)
							yyy=2*height-yyy-1;

						for (int xx = -radius_x; xx <= radius_x; xx++)
						{
							int xxx=xx+x;
							if (xxx<0)
								xxx=abs(xxx)-1;
							if (xxx>=width)
								xxx=2*width-xxx-1;

							if ( (( (*t_input)( yyy * src_stride_h + xxx * src_stride_w + p * src_stride_p ) > region_mean) && !above)
							   || (((*t_input)( yyy * src_stride_h + xxx * src_stride_w + p * src_stride_p ) < region_mean) && above) )
							{	
                (*m_kernel_weighed)(yy+radius_y, xx+radius_x) = 0.;
							}
						}
					}

					// Normalize kernel
					double weighed_sum=0.;
					for (int yy = -radius_y; yy <= radius_y; yy++)
					{
						for (int xx = -radius_x; xx <= radius_x; xx++)
						{
							weighed_sum+= (*m_kernel_weighed)(yy+radius_y, xx+radius_x);
						}
					}
			
					for (int yy = -radius_y; yy <= radius_y; yy++)
					{
						for (int xx = -radius_x; xx <= radius_x; xx++)
						{
              (*m_kernel_weighed)(yy+radius_y, xx+radius_x) /= weighed_sum;
						}
					}

					// Apply the kernel for the <y, x> pixel
					double sum = 0.0;
					for (int yy = -radius_y; yy <= radius_y; yy++)
					{
						// mirror interpolation
						int yyy=yy+y;
						if (yyy<0)
							yyy=abs(yyy)-1;
						if (yyy>=height)
							yyy=2*height-yyy-1;
					
						for (int xx = -radius_x; xx <= radius_x; xx++)
						{
							// mirror interpolation
							int xxx=xx+x;
							if (xxx<0)
								xxx=abs(xxx)-1;
							if (xxx>=width)
								xxx=2*width-xxx-1;

							sum +=  (*m_kernel_weighed)( yy+radius_y, xx+radius_x) * 
								(*t_input)( yyy * src_stride_h + xxx * src_stride_w + p * src_stride_p);
						}
					}
					delete m_kernel_weighed;
				
					// Update output using the FixI macro (round double value)
					(*t_output)(y,x,p) = FixI(sum);
				
				}
				// Regular Gaussian kernel for MSR
				else
				{	
					// Apply the kernel for the <y, x> pixel
					double sum = 0.0;
					int yyy, xxx;
					for (int yy = -radius_y; yy <= radius_y; yy++ )
					{
						// mirror interpolation
						yyy=yy+y;
						if (yyy<0)
							yyy=abs(yyy)-1;
						if (yyy>=height)
							yyy=2*height-yyy-1;
				
						for (int xx = -radius_x; xx <= radius_x; xx++ )
						{
							// mirror interpolation
							xxx=xx+x;
							if (xxx<0)
								xxx=abs(xxx)-1;
							if (xxx>=width)
								xxx=2*width-xxx-1;

							sum +=  (*m_kernel)(yy + radius_y, xx + radius_x ) * (*t_input)( yyy * src_stride_h + xxx * src_stride_w + p * src_stride_p );
						}
					}
					// Update output using the FixI macro (round double value)
					(*t_output)(y,x,p) = FixI(sum);	
				}
			}
		}
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////

}

