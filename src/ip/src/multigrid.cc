#include "ip/multigrid.h"
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
    // TODO: check correctness of dimension between input and ouput (factor 2)
  	const int height_out = restricted.size(0);
		const int width_out = restricted.size(1);
		const int n_planes_out = restricted.size(2);

    for (int p = 0; p < n_planes_out; p ++) 
      for (int y = 0; y < height_out; y ++)
        for (int x = 0; x < width_out; x ++)
        {
				  // on the boundary: copy the corresponding pixel (injection)
					if( (x==0) || (x==width_out-1) || (y==0) || (y==width_out-1))
						restricted(y,x,p) = data(2*y,2*x,p);	
					// full weighting scheme (interior points)
					else
					{
            restricted(y,x,p) = 0.25*data(2*y,2*x,p) + 
              0.125  * ( data(2*y,2*x-1,p)+data(2*y-1,2*x,p)+data(2*y,2*x+1,p)+data(2*y+1,2*x,p) ) +
              0.0625 * ( data(2*y-1,2*x-1,p)+data(2*y+1,2*x-1,p)+data(2*y-1,2*x+1,p)+data(2*y+1,2*x+1,p) );
					}
				}

	}
    

	void project(const DoubleTensor& data, DoubleTensor& projected )
	{
    // TODO: check correctness of dimension between input and ouput (factor 2)
 		const int height_out = projected.size(0);
		const int width_out = projected.size(1);
		const int n_planes_out = projected.size(2);

    for (int p = 0; p < n_planes_out; p ++) 
    {
			// fill in even column and even row
      for (int y = 0; y < height_out; y+=2)
        for (int x = 0; x < width_out; x+=2)
  				projected(y,x,p) = data(y/2, x/2, p);

			// fill in even column/odd row (except boundary)
      for (int y = 1; y < height_out-1; y+=2)
        for (int x = 0; x < width_out; x+=2)
          projected(y,x,p) = 0.5 * ( projected(y+1,x,p) + projected(y-1,x,p) );
				
			// fill in odd column
      for (int y = 0; y < height_out; y++)
        for (int x = 1; x < width_out-1; x+=2)
          projected(y,x,p) = 0.5 * ( projected(y,x+1,p) + projected(y,x-1,p) );

			// Fill in boundaries
			for(int x = 0; x < width_out; x++)
        projected(height_out-1, x, p) = projected(height_out-2, x, p);

			for(int y = 0; y < height_out; y++)
        projected(y, width_out-1, p) = projected(y, width_out-2, p);
		}
	}


	void buildOperator( DoubleTensor& matrix, DoubleTensor& rho, const double lambda, const int type, const DoubleTensor& image) 
	{
		// TODO: check that the dimensions are correct (square matrix)
		const int width_matrix = matrix.size(1);

		const int height = image.size(0);
		const int width = image.size(1);	
	
		matrix.fill(0.);

		for(int i=0; i<width_matrix; i++)
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

			matrix(i, i) = 1.;
			
			// compute the diffusion coefficients associated to the current pixel (i)
			int x_coord = i % width;
			int y_coord = i / width;
			computeCoeff(rho, image, x_coord, y_coord, type);

			// upper diagonal
			if (right)
			{
				matrix(i+1,i) = -lambda*rho(RIGHT);
				matrix(i,i) += lambda*rho(RIGHT);
			}

			// lower diagonal
			if (left)
			{
				matrix((i-1),i) = -lambda*rho(LEFT);
				matrix(i,i) += lambda*rho(LEFT);
			}

			// upper fringe
			if (down)
			{
				matrix(i+width,i) = -lambda*rho(DOWN);
				matrix(i,i) += lambda*rho(DOWN);
			}

			// lower fringe
			if (up)
			{
				matrix(i-width,i) = -lambda*rho(UP);
				matrix(i,i) += lambda*rho(UP);
			}
		}
	} 

  
	void computeCoeff(DoubleTensor& rho, const DoubleTensor& image, const int x_coord, const int y_coord, const int type )
	{
		const int height = image.size(0);
		const int width = image.size(1);

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
		rho.fill(0.);

		// compute the five points stencil coefficients
		if(up) 
			switch(type) 
			{
				case 0:
					rho(UP) = 1.;
					break;
				case 1:
					weber(rho, image, x_coord, y_coord, "up");
				break;
			} 

		if(down)
			switch(type) 
			{
				case 0:
					rho(DOWN) = 1.;
					break;
				case 1:
					weber(rho, image, x_coord, y_coord, "down");
				break;
			} 

		if (left) 
			switch(type) 
			{
				case 0:
					rho(LEFT) = 1.;
					break;
				case 1:
					weber(rho, image, x_coord, y_coord, "left");
				break;
			} 

		if (right) 
			switch(type) 
			{
				case 0:
					rho(RIGHT) = 1.;
					break;
				case 1:
					weber(rho, image, x_coord, y_coord, "right");
				break;
			} 

		rho(CENTER) =  rho(UP) + rho(DOWN) + rho(LEFT) + rho(RIGHT);
	}


	void gaussSeidel(DoubleTensor& result, const DoubleTensor& source, DoubleTensor &rho, const double lambda, const int type) 
	{
		const int height = source.size(0);
		const int width = source.size(1);

		const int res_height = result.size(0);
		const int res_width = result.size(1);

		// check that the dimensions are correct (cmp src and res)
		if ( height != res_height || width != res_width )
		{
			error("gaussSeidel(): Result and source do not have the same dimensions.");
		}

		double up, down, left, right, center;

		// RED-BLACK GAUSS-SEIDEL
    for (int y = 0; y < height; y ++)
    {
     	for (int x = 0; x < width; x ++) 
      {
				// if we are  on the boundary
				if ( (y == 0) || (y == height-1) ||  (x == 0) || (x == width-1) )
				{ 
					// set the boundary values to zero
					result(y,x,0) = 0.;
				} 
				else
				{
					// if we are on an "even" pixel
					int idx = x + y*width;
					if (idx % 2 != 0) 
					{
						computeCoeff(rho, source, x, y, type);

						up = -lambda*rho(UP)* result( y-1, x, 0);
						down = -lambda*rho(DOWN)* result( y+1, x, 0);
						left = -lambda*rho(LEFT)* result( y, x-1, 0);
						right = -lambda*rho(RIGHT)* result( y, x+1, 0);
						center = 1 + lambda*rho(UP) + lambda*rho(DOWN) + lambda*rho(LEFT) + lambda*rho(RIGHT);

						result(y,x,0) = source(y,x,0) / center - (up + down + left + right)/center;
					}
				}
			}
		}

    for (int y = 0; y < height; y ++)
    {
    	for (int x = (y==0?1:0); x < width; x ++ )
      {
				// if we are on the boundary
				if ( (y == 0) || (y == height-1) ||  (x == 0) || (x == width-1) )
				{  
					// set the boundary values to zero
					result(y,x,0) = 0.;
				} 
				else 
				{
					int idx = x+y*width;
					// if we are on an "odd" pixel
					if (idx % 2 == 0) 
					{
						computeCoeff(rho, source, x, y, type);

						up = -lambda*rho(UP)* result( y-1, x, 0);
						down = -lambda*rho(DOWN)* result( y+1, x, 0);
						left = -lambda*rho(LEFT)* result( y, x-1, 0);
						right = -lambda*rho(RIGHT)* result( y, x+1, 0);
						center = 1 + lambda*rho(UP) + lambda*rho(DOWN) + lambda*rho(LEFT) + lambda*rho(RIGHT);

						result(y,x,0) = source(y,x,0) / center - (up + down + left + right)/center;
					}
				}
			}
		}
	}
    
  

	void myMultiply(const DoubleTensor& data, DoubleTensor& result, DoubleTensor& rho, const double lambda, const int type ) 
	{
		// TODO: check that the dimensions are correct (cmp src and res)
		const int height = data.size(0);
		const int width = data.size(1);

    for (int y = 0; y < height; y++)
      for (int x = 0; x < width; x++)
      {
				// BEWARE --- first check where we are
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
					result(y,x,0) = 0.;
				else 
				{
					computeCoeff(rho, data, x, y, type);

					if(up)
					{
						coeff_up = -lambda*rho(UP) * data( y-1, x, 0);
						coeff_center += lambda*rho(UP);
					}
					if(down) 
					{
						coeff_down = -lambda*rho(DOWN) * data( y+1, x, 0);
						coeff_center += lambda*rho(DOWN);
					}
					if(left) 
					{
						coeff_left = -lambda*rho(LEFT) * data( y, x-1, 0);
						coeff_center += lambda*rho(LEFT);
					}
					if(right) 
					{
						coeff_right = -lambda*rho(RIGHT) * data( y, x+1, 0);
						coeff_center += lambda*rho(RIGHT);
					}
	
					result(y,x,0) = coeff_center * data(y,x,0) + coeff_up + coeff_down + coeff_left + coeff_right; 
				}
			}

	}


	void jacobi(DoubleTensor& result, const DoubleTensor& source, DoubleTensor& rho, const double lambda, const int type )
	{ 
		// TODO: check that the dimensions are correct (cmp src and res)
		const int height = source.size(0);
		const int width = source.size(1);

		double up, down, left, right, center;

		DoubleTensor old( result.size(0), result.size(1), result.size(2) );
		old.copy( &result );

		double weight = 1; // weight used in damped Jacobi (set to 1 = pure Jacobi)

    for (int y = 0; y < height; y++)
      for (int x = 0; x < width; x++ )
      {
				// if we are  on the boundary
				if ((y == 0) || (y == height-1) ||  (x == 0) || (x == width-1)) 
					// set the boundary values to zero
					result(y,x,0) = 0.;
				else 
				{
					computeCoeff(rho, source, x, y, type);

					up = -lambda*rho(UP) * old( y-1, x, 0);
					down = -lambda*rho(DOWN) * old( y+1, x, 0);
					left = -lambda*rho(LEFT) * old( y, x-1, 0);
					right = -lambda*rho(RIGHT) * old( y, x+1, 0);
					center = 1 + lambda*rho(UP) + lambda*rho(DOWN) + lambda*rho(LEFT) + lambda*rho(RIGHT);
	
					result(y,x,0) = (1-weight) * old( y, x, 0) + weight * ( source(y,x,0) /center - (up + down + left + right)/center);
				}
			} 

	}


	void weber(DoubleTensor& rho, const DoubleTensor &image, const int x_coord, const  int y_coord, const char *position) 
	{
		if (strcmp(position, "up") == 0) 
		{
			double center = image( y_coord, x_coord, 0); 
			double up = image( y_coord-1, x_coord, 0);
			double diff = fabs(center - up);
			double min = 0.0;

			if (up > center)
				min = center;
			else 
				min = up;

			if (IS_NEAR(diff, 0, 1))
				rho(UP) = min;
			else 
				rho(UP) = min/diff;
		}

		if (strcmp(position, "down") == 0) 
		{
			double center = image( y_coord, x_coord, 0); 
			double down = image( y_coord+1, x_coord, 0);
			double diff = fabs(center - down);
			double min = 0.0;

			if (down > center)
				min = center; 
			else 
				min = down;

			if (IS_NEAR(diff, 0, 1))
				rho(DOWN) = min;
			else 
				rho(DOWN) = min/diff;
		}

		if (strcmp(position, "left") == 0) 
		{
			double center = image( y_coord, x_coord, 0); 
			double left = image( y_coord, x_coord-1, 0);
			double diff = fabs(center - left);
			double min = 0.0;

			if (left > center)
				min = center; 
			else 
				min = left;

			if (IS_NEAR(diff, 0, 1))
				rho(LEFT) = min;
			else 
				rho(LEFT) = min/diff;
		}

		if (strcmp(position, "right") == 0) 
		{
			double center = image( y_coord, x_coord, 0); 
			double right = image( y_coord, x_coord+1, 0);
			double diff = fabs(center - right);
			double min = 0.0;

			if (right > center)
				min = center; 
			else 
				min = right;

			if (IS_NEAR(diff, 0, 1))
				rho(RIGHT) = min;
			else 
				rho(RIGHT) = min/diff;
		}
	}

}

