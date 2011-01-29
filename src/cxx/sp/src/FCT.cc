/**
 * @file src/cxx/sp/src/FCT.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Implement a blitz-based Fast Cosine Transform using FCTPACK functions
 */

#include "sp/FCT.h"

// Declaration of FORTRAN functions from FCTPACK4.1
extern "C" void cosqi_( int *n, double *wsave);
extern "C" void cosqf_( int *n, double *x, double *wsave);
extern "C" void cosqb_( int *n, double *x, double *wsave);


namespace Torch {

  namespace sp {

    /**
     * @brief Compute the 1D FCT of a 1D blitz array
     */
    blitz::Array<double,1> fct(const blitz::Array<double,1>& A)
    {
      // Make a copy of the blitz array. This will allocate a new memory area
      // which will be used to stored the result of the FCT.
      blitz::Array<double,1> res = blitz::copySafedata(A);

      // Declare variables for Lapack
      int n_input = res.extent(0);
      // According to Lapack documentation, the work array must be 
      // dimensioned at least 3*n+15, n being the size of the signal to 
      // process.
      int n_wsave = 3 * n_input + 15;
      double *wsave = new double[n_wsave];
      // Initialize the work array (with exp coefficients, etc.)
      cosqi_( &n_input, wsave);

      // Compute the FCT
      cosqb_( &n_input, res.data(), wsave);
 
      double sqrt2N = sqrt(2./n_input);
      for(int i=0; i<n_input; ++i) 
        res(i) = res(i)/4*(i==0?sqrt(1./n_input):sqrt2N);

      // Deallocate work array
      delete [] wsave;

      return res;
    }

    /**
     * @brief Compute the 1D inverse FCT of a 1D blitz array
     */
    blitz::Array<double,1> ifct(const blitz::Array<double,1>& A)
    {
      // Make a copy of the blitz array. This will allocate a new memory area
      // which will be used to stored the result of the FCT.
      blitz::Array<double,1> res = copySafedata(A);

      // Declare variables for Lapack
      int n_input = res.extent(0);
      // According to Lapack documentation, the work array must be 
      // dimensioned at least 3*n+15, n being the size of the signal to 
      // process.
      int n_wsave = 3 * n_input + 15;
      double *wsave = new double[n_wsave];
      // Initialize the work array (with exp coefficients, etc.)
      cosqi_( &n_input, wsave);

      double sqrt2N = sqrt(2.*n_input);
      for(int i=0; i<n_input; ++i) 
        res(i) = res(i)/(i==0?sqrt(n_input):sqrt2N);

      // Compute the FCT
      cosqf_( &n_input, res.data(), wsave);

      // Deallocate work array
      delete [] wsave;

      return res;
    }


    /**
     * @brief Compute the 2D FCT of a 2D blitz array
     */
    blitz::Array<double,2> fct(const blitz::Array<double,2>& A)
    {
      // Make a copy of the blitz array. This will allocate a new memory area
      // which will be used to stored the result of the FCT.
      blitz::Array<double,2> res(A.extent(0), A.extent(1));

      // Declare variables for Lapack
      int n_input_H = res.extent(0);
      int n_input_W = res.extent(1);

      // Precompute multiplicative factors
      double sqrt1H=sqrt(1./n_input_H);
      double sqrt2H=sqrt(2./n_input_H);
      double sqrt1W=sqrt(1./n_input_W);
      double sqrt2W=sqrt(2./n_input_W);

      // According to Lapack documentation, the work array must be 
      // dimensioned at least 3*n+15, n being the size of the signal to 
      // process.
      int n_wsave_H = 3 * n_input_H + 15;
      double *wsave_H = new double[n_wsave_H];
      cosqi_( &n_input_H, wsave_H);
      int n_wsave_W = 3 * n_input_W + 15;
      double *wsave_W = new double[n_wsave_W];
      cosqi_( &n_input_W, wsave_W);

      double *col_tmp = new double[n_input_H];

      // Apply 1D FCT to each column of the 2D array (array(id_row,id_column))
      for(int j=0; j<n_input_W; ++j)
      {
        // Copy the column into the C array
        for( int i=0; i<n_input_H; ++i)
          col_tmp[i] = A(i,j);

        // Compute the FCT of one column
        cosqb_( &n_input_H, col_tmp, wsave_H);

        // Update the column
        for( int i=0; i<n_input_H; ++i)
          res(i,j) = col_tmp[i];
      }

      // Apply 1D FCT to each row of the resulting matrix
      for(int i=0; i<n_input_H; ++i)
      {
        // Compute the FCT of one row
        cosqb_( &n_input_W, &(res.data()[i*n_input_W]), wsave_W);
      }

      // Deallocate memory
      delete [] col_tmp;
      delete [] wsave_H;
      delete [] wsave_W;

      // Rescale the result
      for(int i=0; i<n_input_H; ++i)
        for(int j=0; j<n_input_W; ++j)
          res(i,j) = res(i,j)/16.*(i==0?sqrt1H:sqrt2H)*(j==0?sqrt1W:sqrt2W);

      return res;
    }


    /**
     * @brief Compute the 2D inverse FCT of a 2D blitz array
     */
    blitz::Array<double,2> ifct(const blitz::Array<double,2>& A)
    {
      // Make a copy of the blitz array. This will allocate a new memory area
      // which will be used to stored the result of the FCT.
      blitz::Array<double,2> res(A.extent(0), A.extent(1));

      // Declare variables for Lapack
      int n_input_H = res.extent(0);
      int n_input_W = res.extent(1);

      // Precompute multiplicative factors
      double sqrt1H=sqrt(1./n_input_H);
      double sqrt2H=sqrt(2./n_input_H);
      double sqrt1W=sqrt(1./n_input_W);
      double sqrt2W=sqrt(2./n_input_W);

      // According to Lapack documentation, the work array must be 
      // dimensioned at least 3*n+15, n being the size of the signal to 
      // process.
      int n_wsave_H = 3 * n_input_H + 15;
      double *wsave_H = new double[n_wsave_H];
      cosqi_( &n_input_H, wsave_H);
      int n_wsave_W = 3 * n_input_W + 15;
      double *wsave_W = new double[n_wsave_W];
      cosqi_( &n_input_W, wsave_W);

      double *col_tmp = new double[n_input_H];

      // Apply 1D inverse FCT to each column of the 2D array (array(id_row,id_column))
      for(int j=0; j<n_input_W; ++j)
      {
        // Copy the column into the C array
        for( int i=0; i<n_input_H; ++i)
          col_tmp[i] = A(i,j)*16/(i==0?sqrt1H:sqrt2H)/(j==0?sqrt1W:sqrt2W);

        // Compute the inverse FCT of one column
        cosqf_( &n_input_H, col_tmp, wsave_H);

        // Update the column
        for( int i=0; i<n_input_H; ++i)
          res(i,j) = col_tmp[i];
      }

      // Apply 1D FCT to each row of the resulting matrix
      for(int i=0; i<n_input_H; ++i)
      {
        // Compute the FCT of one row
        cosqf_( &n_input_W, &(res.data()[i*n_input_W]), wsave_W);
      }

      // Deallocate memory
      delete [] col_tmp;
      delete [] wsave_H;
      delete [] wsave_W;

      // Rescale the result by the size of the input 
      // (as this is not performed by Lapack)
      double norm_factor = 16*n_input_W*n_input_H;
      res /= norm_factor;

      return res;
    }

  }
}

