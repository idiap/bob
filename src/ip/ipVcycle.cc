#include "ipVcycle.h"
#include "multigrid.h"
#include "ipRescaleGray.h"

#ifdef USE_CBLAS
#include "clapack.h"
#endif

//*******************************************************************
//
// This file implements the multigrid V-cycle algorithm from 
// "a multigrid tutorial"[Briggs] 
//
// ---> author: Guillaume Heusch (heusch@idiap.ch)            <---
// ---> author: Laurent El Shafey (laurent.elshafey@idiap.ch) <---
//
//*******************************************************************

#ifdef USE_CBLAS
extern "C" int clapack_dgesv(const enum CBLAS_ORDER Order, const int N, const int NRHS,
		double *A, const int lda, int *ipiv,
		double *B, const int ldb);
#endif

namespace Torch {



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

	const short* src = t_input->t->storage->data + t_input->t->storageOffset;
	short* dst = t_output->t->storage->data + t_output->t->storageOffset;

	const int src_stride_h = t_input->t->stride[0];	// height
	const int src_stride_w = t_input->t->stride[1];	// width

	// An index for the 3D tensor is: [y * stride_h + x * stride_w + p * stride_p]

	const int height = input.size(0);
	const int width = input.size(1);

	height_ = height;
	width_ = width;


	// Tensor to store the diffusion coefficients for each processed pixel

	// Tensor to store the non-rescaled double output
	DoubleTensor* t_output_double = new DoubleTensor( height, width, 1 );
	double* dst_double = t_output_double->t->storage->data + t_output_double->t->storageOffset;

	const int dst_stride_h = t_output_double->t->stride[0];	// height
	const int dst_stride_w = t_output_double->t->stride[1];	// width

        // Fill with 0 the output image (to clear boundaries)
        t_output->fill(0);
        
	// the original image
	DoubleTensor* image_grid = new DoubleTensor(height,  width, 1);

	// take zero as an initial guess
	DoubleTensor *guess = new DoubleTensor( height, width, 1);
	guess->fill(0.);


	//copy the data into the finest image grid
	image_grid->copy( t_input );

	// call to Multigrid V-cycle
	DoubleTensor* light = mgv(*guess, *image_grid, lambda, 0, type );
	
	double* img = image_grid->t->storage->data + image_grid->t->storageOffset;	
	double* lig = light->t->storage->data + light->t->storageOffset;	

	// build final result (R = I/L)
	for (int y=0; y<height; y++)
	{
		const double* lig_row = &lig[ y * src_stride_h ];
		const double* img_row = &img[ y * src_stride_h ];
		double* dst_double_row = &dst_double[ y * dst_stride_h ];
		for (int x=0; x<width; x++, lig_row+=dst_stride_w, dst_double_row+=dst_stride_w, img_row+=dst_stride_w )
		{
			if ( (y==0) || (y == (height - 1)) ||  (x == 0) || (x == width-1) ) 
				*dst_double_row = 1.;
			else 
			{
				if ( IS_NEAR( *lig_row , 0.0 , 0.01 ) ) 
					*dst_double_row = 1.;
				else
					*dst_double_row = *img_row / *lig_row;
			}
		}
	}

	// display purpose
	bool b1=cutExtremum( *t_output_double, 4, 0 );

	// Rescale the values in [0,255] and copy it into the output Tensor
	ipCore *rescale = new ipRescaleGray();
	CHECK_FATAL(rescale->process( *t_output_double ) == true);
	t_output->copy( &(rescale->getOutput(0)) );
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
	const double* b_vec = t_b->t->storage->data + t_b->t->storageOffset;

	const int stride_h = t_b->t->stride[0];	// height
	const int stride_w = t_b->t->stride[1];	// width

	// An index for the 3D tensor is: [y * stride_h + x * stride_w + p * stride_p]
	const int height = b_v.size(0);
	const int width = b_v.size(1);

	// TODO: Check dimensions
	DoubleTensor* result = new DoubleTensor( height_, width_, 1 );
	double* res = result->t->storage->data + result->t->storageOffset;

	const int res_stride_h = result->t->stride[0];	// height
	const int res_stride_w = result->t->stride[1];	// width

	DoubleTensor* rho = new DoubleTensor(5);

	// if we are on the coarsest level -> solve
	if (level == n_grids_-1) 
	{
		DoubleTensor* diffOperator = new DoubleTensor(width_*height_, width_*height_, 1);
		buildOperator(*diffOperator, *rho, lambda, type, b_v );

		double* d_diffOperator=(double*)diffOperator->dataW();
		double* d_b=(double*)t_b->dataW();
		result->copy(t_b);
		double* d_result=(double*)result->dataW();
		
		// Prepare to use LAPACK function
		IntTensor ipiv(width_*height_);
		int info = 0;
#ifdef USE_CBLAS
		info = clapack_dgesv(CblasRowMajor, width_*height_, 1, d_diffOperator, width_*height_, (int*)ipiv.dataW(), d_result, width_*height_);
#endif
		if (info != 0) error("ipVCycle: failure with error %d when solving sparse system\n", info);

		// set boundary pixels to zero
		for (int y=0; y<height; y++)
		{
			double* res_row = &res[ y * res_stride_h ];
			for (int x=0; x<width; x++, res_row+=res_stride_w)
			{
				bool is_on_boundary = (x == 0) || (x == width-1) || (y == 0) || (y == width -1);
	
				if (is_on_boundary)
					*res_row = 0.;
			}
		}
		// Clean
		delete diffOperator;
	} 
	else
	{
		// relax (pre-smooth) 
		gaussSeidel(x_v, b_v, *rho, lambda, type);
      
		// Prepare tensor and compute residual
		DoubleTensor temp( height_, width_, 1);
		double* temp_p = temp.t->storage->data + temp.t->storageOffset;
		const int temp_stride_h = temp.t->stride[0];	// height
		const int temp_stride_w = temp.t->stride[1];	// width

		double* b_v_p = b_v.t->storage->data + b_v.t->storageOffset;
		const int b_v_stride_h = b_v.t->stride[0];	// height
		const int b_v_stride_w = b_v.t->stride[1];	// width

		myMultiply(x_v, temp, *rho, lambda, type);

		for (int y=0; y<height; y++)
		{
			double* temp_row = &temp_p[ y * temp_stride_h];
			double* b_v_row = &b_v_p[  y * b_v_stride_h];
			for (int x=0; x<width; x++, temp_row+=temp_stride_w, b_v_row+=b_v_stride_w )
				*temp_row = *b_v_row - *temp_row;
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

		// Prepare tensor for efficient access
		double* x_v_p = x_v.t->storage->data + x_v.t->storageOffset;
		const int x_v_stride_h = x_v.t->stride[0];	// height
		const int x_v_stride_w = x_v.t->stride[1];	// width

		double* xcorr_p = xcorr.t->storage->data + xcorr.t->storageOffset;
		const int xcorr_stride_h = xcorr.t->stride[0];	// height
		const int xcorr_stride_w = xcorr.t->stride[1];	// width

		// update solution
		for (int y=0; y<height_; y++)
		{
			double* res_row = &res[ y * res_stride_h ];
			double* x_v_row = &x_v_p[ y * x_v_stride_h ];
			double* xcorr_row = &xcorr_p[ y * xcorr_stride_h ];
			for (int x=0; x<width_; x++, res_row+=res_stride_w, x_v_row+=x_v_stride_w, xcorr_row+=xcorr_stride_w )
				*res_row = *x_v_row + * xcorr_row;
		}

		// relax (post-smooth)
		gaussSeidel(*result, b_v, *rho, lambda, type);
	} 

	delete rho;

	return result;
}


bool ipVcycle::cutExtremum(DoubleTensor& data, int distribution_width, int p) 
{
	DoubleTensor* t_data = (DoubleTensor*)&data;	
	double* dat = t_data->t->storage->data + t_data->t->storageOffset;
 
	const int stride_h = t_data->t->stride[0];	// height
	const int stride_w = t_data->t->stride[1];	// width
	const int stride_p = t_data->t->stride[2];	// no planes

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

