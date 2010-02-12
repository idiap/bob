#include "multigrid.h"
#include "string.h"

//*************************************************************************************
//
// This file implements all the functions needed in mutigrid framework
// 
// -> restriction: fine-to-coarse projection (full weighting) 
// -> projection: coarse-to-fine projection (bilinear interpolation) 
// -> build differential operator
// -> compute diffusion coefficients for each pixel 
// -> customized matrix multiplication (takes advantage of the sparse structure)
// -> Relaxation: - Gauss-Seidel
//                - Jacobi 
//  
// 
// ---> author: Guillaume Heusch (heusch@idiap.ch)             <---
// ---> author: Laurent El Shafey (laurent.el-shafey@idiap.ch) <---
//
//************************************************************************************


namespace Torch {


	void restriction(const DoubleTensor& data, DoubleTensor& restricted )
	{
	        const DoubleTensor* t_data = (DoubleTensor*)&data;
        	DoubleTensor* t_restricted = (DoubleTensor*)&restricted;

	        const double* src = t_data->t->storage->data + t_data->t->storageOffset;
	        double* dst = t_restricted->t->storage->data + t_restricted->t->storageOffset;

	        const int stride_h = t_data->t->stride[0];     // height
	        const int stride_w = t_data->t->stride[1];     // width
	        const int stride_p = t_data->t->stride[2];     // no planes
		
	        const int stride_h_out = t_restricted->t->stride[0];     // height
	        const int stride_w_out = t_restricted->t->stride[1];     // width
	        const int stride_p_out = t_restricted->t->stride[2];     // no planes

	        // An index for the 3D tensor is: [y * stride_h + x * stride_w + p * stride_p]
	        // TODO: check correctness of dimension between input and ouput (factor 2)
		const int height = data.size(0);
		const int width = data.size(1);
		const int n_planes = data.size(2);

  		const int height_out = restricted.size(0);
		const int width_out = restricted.size(1);
		const int n_planes_out = restricted.size(2);

	        for (int p = 0; p < n_planes_out; p ++) 
        	{
                	const double* src_plane = &src[p * stride_p];
	                double* dst_plane = &dst[p * stride_p_out];

	                for (int y = 0; y < height_out; y ++)
        	        {
                        	for (int x = 0; x < width_out; x ++)
	                        {
					// on the boundary: copy the corresponding pixel (injection)
					if( (x==0) || (x==width_out-1) || (y==0) || (y==width_out-1))
						dst_plane[x*stride_w_out + y*stride_h_out]=src_plane[2*x*stride_w + 2*y*stride_h];		
					// full weighting scheme (interior points)
					else
					{
						int ind=2*x*stride_w + 2*y*stride_h;
						dst_plane[x*stride_w_out + y*stride_h_out]= 0.25*src_plane[ind]
						+ 0.125*(src_plane[ind-stride_w]+src_plane[ind-stride_h]+src_plane[ind+stride_w]+src_plane[ind+stride_h])
						+ 0.0625*(src_plane[ind-stride_w-stride_h]+src_plane[ind-stride_w+stride_h]
								+src_plane[ind+stride_w-stride_h]+src_plane[ind+stride_w+stride_h]);
					}
				}
			}
		}		
	}
    

	void project(const DoubleTensor& data, DoubleTensor& projected )
	{
	        const DoubleTensor* t_data = (DoubleTensor*)&data;
        	DoubleTensor* t_projected = (DoubleTensor*)&projected;

	        const double* src = t_data->t->storage->data + t_data->t->storageOffset;
	        double* dst = t_projected->t->storage->data + t_projected->t->storageOffset;

	        const int stride_h = t_data->t->stride[0];     // height
	        const int stride_w = t_data->t->stride[1];     // width
	        const int stride_p = t_data->t->stride[2];     // no planes
		
	        const int stride_h_out = t_projected->t->stride[0];     // height
	        const int stride_w_out = t_projected->t->stride[1];     // width
	        const int stride_p_out = t_projected->t->stride[2];     // no planes

	        // An index for the 3D tensor is: [y * stride_h + x * stride_w + p * stride_p]
	        // TODO: check correctness of dimension between input and ouput (factor 2)
		const int height = data.size(0);
		const int width = data.size(1);
		const int n_planes = data.size(2);

  		const int height_out = projected.size(0);
		const int width_out = projected.size(1);
		const int n_planes_out = projected.size(2);

	        for (int p = 0; p < n_planes_out; p ++) 
        	{
                	const double* src_plane = &src[p * stride_p];
	                double* dst_plane = &dst[p * stride_p_out];

			// fill in even column and even row
	                for (int y = 0; y < height_out; y+=2)
                        	for (int x = 0; x < width_out; x+=2)
					dst_plane[x*stride_w_out + y*stride_h_out] = src_plane[x/2*stride_w + y/2*stride_h];

			// fill in even column/odd row (except boundary)
	                for (int y = 1; y < height_out-1; y+=2)
                        	for (int x = 0; x < width_out; x+=2)
					dst_plane[x*stride_w_out + y*stride_h_out] = 0.5*(dst_plane[x*stride_w_out + (y+1)*stride_h_out]
						+dst_plane[x*stride_w_out + (y-1)*stride_h_out]);
				
			// fill in odd column
	                for (int y = 0; y < height_out; y++)
                        	for (int x = 1; x < width_out-1; x+=2)
					dst_plane[x*stride_w_out + y*stride_h_out] = 0.5*(dst_plane[(x+1)*stride_w_out + y*stride_h_out]
						+dst_plane[(x-1)*stride_w_out + y*stride_h_out]);

			// Fill in boundaries
			for(int x = 0; x < width_out; x++)
				dst_plane[x*stride_w_out+(height_out-1)*stride_h_out]=dst_plane[x*stride_w_out+(height_out-2)*stride_h_out];

			for(int y = 0; y < height_out; y++)
				dst_plane[(width_out-1)*stride_w_out+y*stride_h_out]=dst_plane[(width_out-2)*stride_w_out+y*stride_h_out];
		}	
	}


	void buildOperator( DoubleTensor& matrix, DoubleTensor& rho, const double lambda, const int type, const DoubleTensor& image) 
	{
	        DoubleTensor* t_matrix = (DoubleTensor*)&matrix;
		//DoubleTensor* t_projected = (DoubleTensor*)&projected;

	        double* mat = t_matrix->t->storage->data + t_matrix->t->storageOffset;
	        //double* dst = t_projected->t->storage->data + t_projected->t->storageOffset;

	        const int stride_h = t_matrix->t->stride[0];     // height
	        const int stride_w = t_matrix->t->stride[1];     // width

		// TODO: check that the dimensions are correct (square matrix)
		const int height_matrix = matrix.size(0);
		const int width_matrix = matrix.size(1);

		const int height = image.size(0);
		const int width = image.size(1);	
	
		t_matrix->fill(0.);

		// Prepare pointer to efficient access to rho
		double* rho_p =  rho.t->storage->data + rho.t->storageOffset;

		for (int i=0; i<width_matrix; i++)
		{
			// CHECK WHERE WE ARE
			bool up = true;
			bool down = true;
			bool right = true;
			bool left = true;

			// we are on the first image line
			if ( i<width )
				up = false;

			// we are on the last image line
			if ( i >= (height-1)*width )
				down = false;

			// we are on the first image column
			if ( (i % width) == 0 )
				left = false;

			// we are on the last image column
			if ( (i % width) == (width - 1) )
				right = false;

			mat[ i*stride_w + i*stride_h] = 1.;
			
			// compute the diffusion coefficients associated to the current pixel (i)
			int x_coord = i % width;
			int y_coord = i / width;
			computeCoeff(rho, image, x_coord, y_coord, type);

			// upper diagonal
			if (right)
			{
				mat[ i*stride_w + (i+1)*stride_h] = -lambda*rho_p[RIGHT];
				mat[ i*stride_w + i*stride_h] += lambda*rho_p[RIGHT];
			}

			// lower diagonal
			if (left)
			{
				mat[ i*stride_w + (i-1)*stride_h] = -lambda*rho_p[LEFT];
				mat[ i*stride_w + i*stride_h] += lambda*rho_p[LEFT];
			}

			// upper fringe
			if (down)
			{
				mat[ i*stride_w + (i+width)*stride_h] = -lambda*rho_p[DOWN];
				mat[ i*stride_w + i*stride_h] += lambda*rho_p[DOWN];
			}

			// lower fringe
			if (up)
			{
				mat[ i*stride_w + (i-width)*stride_h] = -lambda*rho_p[UP];
				mat[ i*stride_w + i*stride_h] += lambda*rho_p[UP];
			}
		}
	} 

  
	void computeCoeff(DoubleTensor& rho, const DoubleTensor& image, const int x_coord, const int y_coord, const int type )
	{
	        DoubleTensor* t_image = (DoubleTensor*)&image;
		DoubleTensor* t_rho = (DoubleTensor*)&rho;

	        double* img = t_image->t->storage->data + t_image->t->storageOffset;
	        double* rho_p = t_rho->t->storage->data + t_rho->t->storageOffset;

	        const int stride_h = t_image->t->stride[0];     // height
	        const int stride_w = t_image->t->stride[1];     // width
	        const int stride_p = t_image->t->stride[2];     // no planes

		const int height = image.size(0);
		const int width = image.size(1);
		const int n_planes = image.size(2);

		bool up = true;
		bool down = true;
		bool left = true;
		bool right = true;

		// we are on the first image line
		if (y_coord == 0 )
			up = false;
    
		// we are on the last image line    
		if (y_coord == height-1 ) 
			down = false;
    
		// we are on the first image column
		if (x_coord == 0 )
			left = false;
    
		// we are on the last image column
		if (x_coord == width-1 )
			right = false;
    
		// initial contrast values set to zero, updated if we can compute it ...
		t_rho->fill(0.);

		// compute the five points stencil coefficients
		if (up) 
			switch(type) 
			{
				case 0:
					rho_p[UP] = 1.;
					break;
				case 1:
					weber(rho, image, x_coord, y_coord, "up");
				break;
			} 

		if (down) 
			switch(type) 
			{
				case 0:
					rho_p[DOWN] = 1.;
					break;
				case 1:
					weber(rho, image, x_coord, y_coord, "down");
				break;
			} 

		if (left) 
			switch(type) 
			{
				case 0:
					rho_p[LEFT] = 1.;
					break;
				case 1:
					weber(rho, image, x_coord, y_coord, "left");
				break;
			} 

		if (right) 
			switch(type) 
			{
				case 0:
					rho_p[RIGHT] = 1.;
					break;
				case 1:
					weber(rho, image, x_coord, y_coord, "right");
				break;
			} 

		rho_p[CENTER] =  rho_p[UP] + rho_p[DOWN] + rho_p[LEFT] + rho_p[RIGHT];
	}


	void gaussSeidel(DoubleTensor& result, const DoubleTensor& source, DoubleTensor &rho, const double lambda, const int type) 
	{
	        DoubleTensor* t_result = (DoubleTensor*)&result;
		DoubleTensor* t_source = (DoubleTensor*)&source;
		DoubleTensor* t_rho = (DoubleTensor*)&rho;

	        double* res = t_result->t->storage->data + t_result->t->storageOffset;
	        double* src = t_source->t->storage->data + t_source->t->storageOffset;
		double* rho_p = t_rho->t->storage->data + t_rho->t->storageOffset;

	        const int src_stride_h = t_source->t->stride[0];     // height
	        const int src_stride_w = t_source->t->stride[1];     // width
	        const int src_stride_p = t_source->t->stride[2];     // no planes

	        const int res_stride_h = t_result->t->stride[0];     // height
	        const int res_stride_w = t_result->t->stride[1];     // width
	        const int res_stride_p = t_result->t->stride[2];     // no planes
	
		const int rho_stride = t_rho->t->stride[0];

		const int height = source.size(0);
		const int width = source.size(1);

		const int res_height = result.size(0);
		const int res_width = result.size(1);

		// check that the dimensions are correct (cmp src and res)
		if ( height != res_height || width != res_width )
		{
			error("gaussSeidel(): Result and source do not have the same dimensions.");
		}

    		int dim = width*height;
		double up, down, left, right, center;

		// RED-BLACK GAUSS-SEIDEL
                for (int y = 0; y < height; y ++)
       	        {
			double* src_row = &src[ y * src_stride_h ];
			double* res_row = &res[ y * src_stride_h ];
                       	for (int x = 0; x < width; x ++, src_row+=src_stride_w, res_row+=res_stride_w ) 
                        {
				// if we are  on the boundary
				if ( (y == 0) || (y == height-1) ||  (x == 0) || (x == width-1) )
				{ 
					// set the boundary values to zero
					*res_row = 0.;
				} 
				else 
				{
					// if we are on an "even" pixel
					int idx = x + y*width;
					if (idx % 2 != 0) 
					{
						computeCoeff(rho, source, x, y, type);

						up = -lambda*rho_p[UP]* res[ (y-1)*res_stride_h + x*res_stride_w ];
						down = -lambda*rho_p[DOWN]* res[ (y+1)*res_stride_h + x*res_stride_w ];
						left = -lambda*rho_p[LEFT]* res[ y*res_stride_h + (x-1)*res_stride_w ];
						right = -lambda*rho_p[RIGHT]* res[ y*res_stride_h + (x+1)*res_stride_w ];
						center = 1 + lambda*rho_p[UP] + lambda*rho_p[DOWN] + lambda*rho_p[LEFT] + lambda*rho_p[RIGHT];

						*res_row = *src_row / center - (up + down + left + right)/center;
					}
				}
			}
		}

                for (int y = 0; y < height; y ++)
       	        {
			double* src_row = &src[ y * src_stride_h ];
			double* res_row = &res[ y * src_stride_h ];
                       	for (int x = (y==0?1:0); x < width; x ++, src_row+=src_stride_w, res_row+=res_stride_w )
                        {
				// if we are on the boundary
				if ( (y == 0) || (y == height-1) ||  (x == 0) || (x == width-1) )
				{  
					// set the boundary values to zero
					*res_row = 0.;
				} 
				else 
				{
					int idx = x+y*width;
					// if we are on an "odd" pixel
					if (idx % 2 == 0) 
					{
						computeCoeff(rho, source, x, y, type);

						up = -lambda*rho_p[UP]* res[ (y-1)*res_stride_h + x*res_stride_w ];
						down = -lambda*rho_p[DOWN]* res[ (y+1)*res_stride_h + x*res_stride_w ];
						left = -lambda*rho_p[LEFT]* res[ y*res_stride_h + (x-1)*res_stride_w ];
						right = -lambda*rho_p[RIGHT]* res[ y*res_stride_h + (x+1)*res_stride_w ];
						center = 1 + lambda*rho_p[UP] + lambda*rho_p[DOWN] + lambda*rho_p[LEFT] + lambda*rho_p[RIGHT];

						*res_row = *src_row / center - (up + down + left + right)/center;
					}
				}
			}
		}
	}
    
  

	void myMultiply(const DoubleTensor& data, DoubleTensor& result, DoubleTensor& rho, const double lambda, const int type ) 
	{

	        DoubleTensor* t_result = (DoubleTensor*)&result;
		const DoubleTensor* t_data = (DoubleTensor*)&data;

	        double* res = t_result->t->storage->data + t_result->t->storageOffset;
	        const double* src = t_data->t->storage->data + t_data->t->storageOffset;

	        const int src_stride_h = t_data->t->stride[0];     // height
	        const int src_stride_w = t_data->t->stride[1];     // width

	        const int res_stride_h = t_result->t->stride[0];     // height
	        const int res_stride_w = t_result->t->stride[1];     // width

		// TODO: check that the dimensions are correct (cmp src and res)
		const int height = data.size(0);
		const int width = data.size(1);

	        double* rho_p = rho.t->storage->data + rho.t->storageOffset;

                for (int y = 0; y < height; y ++)
       	        {
			double* res_row=&res[ y * res_stride_h ];
			const double* src_row=&src[ y * src_stride_h ];
                       	for (int x = 0; x < width; x++, src_row+=src_stride_w, res_row+=res_stride_w )
                        {

				// BEWARE --- first check were we are
				bool up = true;
				bool down = true;
				bool left = true;
				bool right = true;
    
				double coeff_up = 0.0;
				double coeff_down = 0.0;
				double coeff_left = 0.0;
				double coeff_right = 0.0;
				double coeff_center = 1.0;
    
				// we are on the first image line
				if (y == 0)
					up = false;

				// we are on the last image line    
				if (y == height-1 ) 
					down = false;
    
				// we are on the first image column
				if (x  == 0)
					left = false;
    
				// we are on the last image column
				if (x == width-1 )
					right = false;

				bool is_on_boundary = (!up || !down || !left || !right);
	
				if (is_on_boundary)
					*res_row = 0.;//result.set(y, x, 0, 0.);
				else 
				{
					int i=x+y*width;
					computeCoeff(rho, data, x, y, type);

					if (up)
					{
						coeff_up = -lambda*rho_p[UP]* src[ (y-1)*src_stride_h + x*src_stride_w];
						coeff_center += lambda*rho_p[UP];
					}
					if (down) 
					{
						coeff_down = -lambda*rho_p[DOWN]* src[ (y+1)*src_stride_h + x*src_stride_w];//data.get(y+1,x,0);
						coeff_center += lambda*rho_p[DOWN];
					}
					if (left) 
					{
						coeff_left = -lambda*rho_p[LEFT]*src[ y*src_stride_h + (x-1)*src_stride_w];//data.get(y,x-1,0);
						coeff_center += lambda*rho_p[LEFT];
					}
					if (right) 
					{
						coeff_right = -lambda*rho_p[RIGHT]*src[ y*src_stride_h + (x+1)*src_stride_w];//data.get(y,x+1,0);
						coeff_center += lambda*rho_p[RIGHT];
					}
	
					*res_row= coeff_center* *src_row + coeff_up + coeff_down + coeff_left + coeff_right; 
				}
			}
		}
	}


	void jacobi(DoubleTensor& result, const DoubleTensor& source, DoubleTensor& rho, const double lambda, const int type )
	{ 
    
	        DoubleTensor* t_result = (DoubleTensor*)&result;
		DoubleTensor* t_source = (DoubleTensor*)&source;

	        double* res = t_result->t->storage->data + t_result->t->storageOffset;
	        double* src = t_source->t->storage->data + t_source->t->storageOffset;

	        const int src_stride_h = t_source->t->stride[0];     // height
	        const int src_stride_w = t_source->t->stride[1];     // width

	        const int res_stride_h = t_result->t->stride[0];     // height
	        const int res_stride_w = t_result->t->stride[1];     // width

		// TODO: check that the dimensions are correct (cmp src and res)
		const int height = source.size(0);
		const int width = source.size(1);
		const int n_planes = source.size(2);

		int dim = width*height;
		double up, down, left, right, center;

		DoubleTensor* old=new DoubleTensor( result.size(0), result.size(1), result.size(2) );
		old->copy( &result );
	        double* old_p = old->t->storage->data + old->t->storageOffset;
	        const int old_stride_h = old->t->stride[0];     // height
	        const int old_stride_w = old->t->stride[1];     // width
 
		double weight = 1; // weight used in damped Jacobi (set to 1 = pure Jacobi)

	        double* rho_p = rho.t->storage->data + rho.t->storageOffset;

                for (int y = 0; y < height; y ++)
       	        {
			double* res_row=&res[ y * res_stride_h ];
			double* src_row=&src[ y * src_stride_h ];
                       	for (int x = 0; x < width; x ++, res_row+=res_stride_w, src_row+=src_stride_w )
                        {
				// if we are  on the boundary
				if ((y == 0) || (y == height-1) ||  (x == 0) || (x == width-1)) 
					// set the boundary values to zero
					*res_row = 0.;
				else 
				{
					int idx= x+y*width;
					computeCoeff(rho, source, x, y, type);

					up = -lambda*rho_p[UP]*old_p[ (y-1) * old_stride_h + x * old_stride_w ];
					down = -lambda*rho_p[DOWN]*old_p[ (y+1) * old_stride_h + x * old_stride_w ];
					left = -lambda*rho_p[LEFT]*old_p[ y * old_stride_h + (x-1) * old_stride_w ];
					right = -lambda*rho_p[RIGHT]*old_p[ y * old_stride_h + (x+1) * old_stride_w ];
					center = 1 + lambda*rho_p[UP] + lambda*rho_p[DOWN] + lambda*rho_p[LEFT] + lambda*rho_p[RIGHT];
	
					*res_row = (1-weight) * old_p[ y * old_stride_h + x * old_stride_w ] + weight*( *src_row /center - (up + down + left + right)/center);
				}
			} 
		}
		delete old;
	}


	void weber(DoubleTensor& rho, const DoubleTensor &image, const int x_coord, const  int y_coord, const char *position) 
	{
	        DoubleTensor* t_image = (DoubleTensor*)&image;
		DoubleTensor* t_rho = (DoubleTensor*)&rho;

	        double* img = t_image->t->storage->data + t_image->t->storageOffset;
	        double* rho_p = t_rho->t->storage->data + t_rho->t->storageOffset;

	        const int stride_h = t_image->t->stride[0];     // height
	        const int stride_w = t_image->t->stride[1];     // width
	        const int stride_p = t_image->t->stride[2];     // no planes

		const int height = image.size(0);
		const int width = image.size(1);
		const int n_planes = image.size(2);
		
		if (strcmp(position, "up") == 0) 
		{
			double center = img[ y_coord * stride_h + x_coord * stride_w ]; 
			double up = img[ (y_coord-1) * stride_h + x_coord * stride_w ];
			double diff = fabs(center - up);
			double min = 0.0;

			if (up > center)
				min = center;
			else 
				min = up;

			if (IS_NEAR(diff, 0, 1))
				rho_p[UP] = min;
			else 
				rho_p[UP] = min/diff;
		}

		if (strcmp(position, "down") == 0) 
		{
			double center = img[ y_coord * stride_h + x_coord * stride_w ]; 
			double down = img[ (y_coord+1) * stride_h + x_coord * stride_w ];
			double diff = fabs(center - down);
			double min = 0.0;

			if (down > center)
				min = center; 
			else 
				min = down;

			if (IS_NEAR(diff, 0, 1))
				rho_p[DOWN] = min;
			else 
				rho_p[DOWN] = min/diff;
		}

		if (strcmp(position, "left") == 0) 
		{
			double center = img[ y_coord * stride_h + x_coord * stride_w ]; 
			double left = img[ y_coord * stride_h + (x_coord-1) * stride_w ];
			double diff = fabs(center - left);
			double min = 0.0;

			if (left > center)
				min = center; 
			else 
				min = left;

			if (IS_NEAR(diff, 0, 1))
				rho_p[LEFT] = min;
			else 
				rho_p[LEFT] = min/diff;
		}

		if (strcmp(position, "right") == 0) 
		{
			double center = img[ y_coord * stride_h + x_coord * stride_w ]; 
			double right = img[ y_coord * stride_h + (x_coord+1) * stride_w ];
			double diff = fabs(center - right);
			double min = 0.0;

			if (right > center)
				min = center; 
			else 
				min = right;

			if (IS_NEAR(diff, 0, 1))
				rho_p[RIGHT] = min;
			else 
				rho_p[RIGHT] = min/diff;
		}
	}

}

