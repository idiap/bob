/**
 * @file cxx/ip/src/ipRelaxation.cc
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
#include "ip/ipRelaxation.h"
#include "ip/multigrid.h"
#include "ip/ipRescaleGray.h"

namespace bob {

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
	// Accept only 3D tensors of bob::Image type
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

	const int height = input.size(0);
	const int width = input.size(1);

	DoubleTensor* rho = new DoubleTensor(5);
	DoubleTensor* image = new DoubleTensor(height,width,1);
	DoubleTensor* light = new DoubleTensor(height,width,1);

  image->copy(t_input);
  light->copy(t_input);

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
  const ShortTensor* out_l = (const ShortTensor*)&rescale->getOutput(0);
	light->copy( out_l );
	delete rescale;
	
	// build final result (R = I/L)
	for(int y = 0 ; y < height ; y++)
	{
		for(int x = 0 ; x < width ; x++ )
		{
			// Set R=I/L equal to 1 at the border
			if ((y == 0) || (y == height - 1) ||  (x == 0) || (x == width-1)) 
				(*light)(y,x,0) = 1.;
			else 
			{
				if (IS_NEAR((*light)(y,x,0), 0.0, 1)) 
					(*light)(y,x,0) = 1.;  
				else
					(*light)(y,x,0) = (*t_input)(y,x,0) / (*light)(y,x,0);
			}
		}
	}
 	cutExtremum(*light, 4); 
       
 
	// Rescale the values in [0,255] and copy it into the output Tensor
	rescale = new ipRescaleGray();
	CHECK_FATAL(rescale->process(*light) == true);
  out_l = (const ShortTensor*)&rescale->getOutput(0);
	t_output->copy( out_l );

	// Clean up
	delete rescale;
  
	delete rho;
	delete image;
	delete light;

	return true;
}


bool ipRelaxation::cutExtremum(DoubleTensor& data, int distribution_width) 
{
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
		for(int x = 0 ; x < width ; x++ )
		{
			mean_out += data(y,x,0);
		}
	}
	mean_out /= wxh;
    
	// compute variance and standard deviation
	for(int y = 0 ; y < height ; y++)
	{
		for(int x = 0 ; x < width ; x++ )
		{
			var_out += ( data(y,x,0) - mean_out ) * ( data(y,x,0) - mean_out );    
		}
	}
	var_out /= (wxh - 1);
	std_dev = sqrt(var_out);

	/// Cut
	double mean_plus_dxstd = mean_out + distribution_width*std_dev;
	double mean_minus_dxstd = mean_out - distribution_width*std_dev;
	
	for(int y = 0 ; y < height ; y++)
	{
		for(int x = 0 ; x < width ; x++ )
		{
			if ( data(y,x,0) > mean_plus_dxstd )
				data(y,x,0) = mean_plus_dxstd;
      
			if ( data(y,x,0) < mean_minus_dxstd )
				data(y,x,0) = mean_minus_dxstd;
		}
	}
	return true;
}


}
