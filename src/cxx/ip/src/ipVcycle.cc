/**
 * @file cxx/ip/src/ipVcycle.cc
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
#include "ip/ipVcycle.h"
#include "ip/multigrid.h"
#include "ip/ipRescaleGray.h"

//*******************************************************************
//
// This file implements the multigrid V-cycle algorithm from 
// "a multigrid tutorial"[Briggs] 
//
// ---> author: Guillaume Heusch (heusch@idiap.ch)            <---
// ---> author: Laurent El Shafey (laurent.elshafey@idiap.ch) <---
//
//*******************************************************************

// Declaration of the external Fortran library. This function performs an eigenvalue decomposition.
extern "C" void dgesv_( int *N, int *NRHS, double *A, int *lda, int *ipiv, 
						double *B, int *ldb, int *info);


namespace bob {



/////////////////////////////////////////////////////////////////////////
// Constructor
ipVcycle::ipVcycle()
	:	ipCore()
{
	addDOption("lambda", 5.0, "Relative importance of the smoothness constraint");
	addIOption("n_grids", 1, "Number of grids used in the v-cycle");
	addIOption("type", 1, "Type of diffusion (coefficients)");
}

/////////////////////////////////////////////////////////////////////////
// Destructor
ipVcycle::~ipVcycle()
{
}


//////////////////////////////////////////////////////////////////////////
// Check if the input tensor has the right dimensions and type

bool ipVcycle::checkInput(const Tensor& input) const
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

bool ipVcycle::allocateOutput(const Tensor& input)
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
	// Accept only gray images
	if (	input.size(2) !=1 )
	{
		warning("ipVcycle::checkInput(): Non gray level image (multiple channels).");
		return false;
	}

	return true;
}


/////////////////////////////////////////////////////////////////////////
// Process some input tensor (the input is checked, the outputs are allocated)

bool ipVcycle::processInput(const Tensor& input)
{
	// Get the parameters
	const double lambda = getDOption("lambda");
	const int n_grids = getIOption("n_grids");
	const int type = getIOption("type");

	n_grids_ = n_grids;

	// Prepare the input and output 3D image tensors
	const ShortTensor* t_input = (ShortTensor*)&input;
	ShortTensor* t_output = (ShortTensor*)m_output[0];

	const int height = input.size(0);
	const int width = input.size(1);

	height_ = height;
	width_ = width;


	// Tensor to store the non-rescaled double output
	DoubleTensor* t_output_double = new DoubleTensor( height, width, 1 );

  // Fill with 0 the output image (to clear boundaries)
  t_output->fill(0);
        
	// the original image
	DoubleTensor* image_grid = new DoubleTensor(height,  width, 1);
	//copy the data into the finest image grid
	image_grid->copy( t_input );

	// take zero as an initial guess
	DoubleTensor *guess = new DoubleTensor( height, width, 1);
	guess->fill(0.);

	// call to Multigrid V-cycle
	DoubleTensor* light = mgv(*guess, *image_grid, lambda, 0, type );	


	// build final result (R = I/L)
	for (int y=0; y<height; y++)
		for (int x=0; x<width; x++ )
		{
			if ( (y==0) || (y == (height - 1)) ||  (x == 0) || (x == width-1) ) 
				(*t_output_double)(y, x, 0) = 1.;
			else 
			{
				if ( IS_NEAR( light->get(y,x,0) , 0.0 , 0.01 ) ) 
					(*t_output_double)(y, x, 0) = 1.;
				else
					(*t_output_double)(y, x, 0) = (*image_grid)(y,x,0) / (*light)(y,x,0);
			}
		}

	// display purpose
	bool b1=cutExtremum( *t_output_double, 4, 0 );
  if( !b1)
    warning("ipVcycle::processInput::cutExtremum() failed");

	// Rescale the values in [0,255] and copy it into the output Tensor
	ipCore *rescale = new ipRescaleGray();
	CHECK_FATAL(rescale->process( *t_output_double ) == true);
  const ShortTensor *out = (const ShortTensor*)&rescale->getOutput(0);
	t_output->copy( out );
	delete rescale;

	// Clean up
	delete guess;
	delete image_grid;
	delete light;
	delete t_output_double;

	return true;
}

DoubleTensor* ipVcycle::mgv(DoubleTensor& x_v, DoubleTensor& b_v, double lambda, int level, int type )
{
	// Prepare the input and output 3D image tensors
	DoubleTensor* t_b = (DoubleTensor*)&b_v;
	const int height = b_v.size(0);
	const int width = b_v.size(1);

	// TODO: Check dimensions
	DoubleTensor* result = new DoubleTensor( height_, width_, 1 );
	DoubleTensor* rho = new DoubleTensor(5);

	// if we are on the coarsest level -> solve
	if (level == n_grids_-1) 
	{
		DoubleTensor* diffOperator = new DoubleTensor(width_*height_, width_*height_);
		buildOperator(*diffOperator, *rho, lambda, type, b_v );

		result->copy(t_b);
		
		int info = 0;
		// Prepare to use LAPACK function
		IntTensor ipiv(width_*height_);
		int N =  width_*height_;
		int lda = N;
		int ldb = N;
		int NRHS = 1;
		double* d_diffOperator=(double*)diffOperator->dataW();
		double* d_result=(double*)result->dataW();
		// Note: In principle, data should be transposed (column-major order instead of row-major order).
		//       Anyway, it is not necessary here with the dgesv_function.
		dgesv_( &N, &NRHS, d_diffOperator, &lda, (int*)ipiv.dataW(), d_result, &ldb, &info );
    diffOperator->resetFromData();
    result->resetFromData();

		if (info != 0) error("ipVCycle: error %d in LAPACK function dgesv_ (Solving a linear system).\n", info);

		// set boundary pixels to zero
		for (int y=0; y<height; y++)
			for (int x=0; x<width; x++)
				if ( (x == 0) || (x == width-1) || (y == 0) || (y == width -1) )
					(*result)(y,x,0) = 0.;

		// Clean
		delete diffOperator;
	} 
	else
	{
		// relax (pre-smooth) 
		gaussSeidel(x_v, b_v, *rho, lambda, type);
      
		// Prepare tensor and compute residual
		DoubleTensor temp( height_, width_, 1);

		myMultiply(x_v, temp, *rho, lambda, type);

		for (int y=0; y<height; y++)
		{
			for (int x=0; x<width; x++ )
				temp(y,x,0) = b_v(y,x,0) - temp(y,x,0);
		}
      
		// restrict residual
		DoubleTensor rhat( height_/2, width_/2,1);
		restriction(temp, rhat);
		width_ /= 2;
		height_ /= 2;

		// recursive call to mgv
		DoubleTensor* xhat;
		DoubleTensor zero(height_, width_, 1);
		zero.fill(0.);
		// xhat memory will be allocated by the recursive call to mgv
		xhat = mgv(zero, rhat, lambda, level+1, type);
      
		// interpolate correction
		DoubleTensor xcorr( height_*2, width_*2, 1); 
		project(*xhat, xcorr);
		width_ *= 2;
		height_ *= 2;

		// free xhat
		delete xhat;

		// update solution
		for (int y=0; y<height_; y++)
		{
			for (int x=0; x<width_; x++ )
				(*result)(y,x,0) = x_v(y,x,0) + xcorr(y,x,0);
		}

		// relax (post-smooth)
		gaussSeidel(*result, b_v, *rho, lambda, type);
	} 

	delete rho;

	return result;
}


bool ipVcycle::cutExtremum(DoubleTensor& data, int distribution_width, int p) 
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
		for(int x = 0 ; x < width ; x++)
			mean_out += data(y,x,0);

	mean_out /= wxh;
    
	// compute variance and standard deviation
	for(int y = 0 ; y < height ; y++)
		for(int x = 0 ; x < width ; x++ )
			var_out += ( data(y,x,0) - mean_out ) * ( data(y,x,0) - mean_out );    

	var_out /= (wxh - 1);
	std_dev = sqrt(var_out);
    
	/// Cut
	double mean_plus_dxstd = mean_out + distribution_width*std_dev;
	double mean_minus_dxstd = mean_out - distribution_width*std_dev;
	
	for(int y = 0 ; y < height ; y++)
		for(int x = 0 ; x < width ; x++ )
		{
			if ( data(y,x,0) > mean_plus_dxstd )
				data(y,x,0) = mean_plus_dxstd;
      
			if ( data(y,x,0) < mean_minus_dxstd )
				data(y,x,0) = mean_minus_dxstd;
		}

	return true;
}
  


}

