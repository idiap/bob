/**
 * @file src/cxx/sp/src/FFT.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Implement a blitz-based FFT using FFTPACK functions
 */

#include "sp/FFT.h"

// To deal with Fotran complex type
typedef struct { double real, imag; } complex_;

// Declaration of FORTRAN functions from FFTPACK4.1
extern "C" void cffti_( int *n, double *wsave);
extern "C" void cfftf_( int *n, complex_ *x, double *wsave);
extern "C" void cfftb_( int *n, complex_ *x, double *wsave);


namespace Torch {

  namespace sp {

    /**
     * @brief Compute the 1D FFT of a 1D blitz array
     */
    blitz::Array<std::complex<double>,1> 
      fft(const blitz::Array<std::complex<double>,1>& A)
    {
      // Make a copy of the blitz array. This will allocate a new memory area
      // which will be used to stored the result of the FFT.
      blitz::Array<std::complex<double>,1> res = blitz::copySafedata(A);

      // Declare variables for Lapack
      int n_input = res.extent(0);
      // According to Lapack documentation, the work array must be 
      // dimensioned at least 4*n+15, n being the size of the signal to 
      // process.
      int n_wsave = 4 * n_input + 15;
      double *wsave = new double[n_wsave];
      // Initialize the work array (with exp coefficients, etc.)
      cffti_( &n_input, wsave);

      // Compute the FFT
      cfftf_( &n_input, reinterpret_cast<complex_*>(res.data()), wsave);

      // Deallocate work array
      delete [] wsave;

      return res;
    }

    /**
     * @brief Compute the 1D inverse FFT of a 1D blitz array
     */
    blitz::Array<std::complex<double>,1> 
      ifft(const blitz::Array<std::complex<double>,1>& A)
    {
      // Make a copy of the blitz array. This will allocate a new memory area
      // which will be used to stored the result of the FFT.
      blitz::Array<std::complex<double>,1> res = copySafedata(A);

      // Declare variables for Lapack
      int n_input = res.extent(0);
      // According to Lapack documentation, the work array must be 
      // dimensioned at least 4*n+15, n being the size of the signal to 
      // process.
      int n_wsave = 4 * n_input + 15;
      double *wsave = new double[n_wsave];
      // Initialize the work array (with exp coefficients, etc.)
      cffti_( &n_input, wsave);

      // Compute the FFT
      cfftb_( &n_input, reinterpret_cast<complex_*>(res.data()), wsave);

      // Deallocate work array
      delete [] wsave;

      // Rescale the result by the size of the input 
      // (as this is not performed by Lapack)
      res /= static_cast<double>(n_input);

      return res;
    }


    /**
     * @brief Compute the 2D FFT of a 2D blitz array
     */
    blitz::Array<std::complex<double>,2> 
      fft(const blitz::Array<std::complex<double>,2>& A)
    {
      // Make a copy of the blitz array. This will allocate a new memory area
      // which will be used to stored the result of the FFT.
      blitz::Array<std::complex<double>,2> res(A.extent(0), A.extent(1));

      // Declare variables for Lapack
      int n_input_H = res.extent(0);
      int n_input_W = res.extent(1);
      // According to Lapack documentation, the work array must be 
      // dimensioned at least 4*n+15, n being the size of the signal to 
      // process.
      int n_wsave_H = 4 * n_input_H + 15;
      double *wsave_H = new double[n_wsave_H];
      cffti_( &n_input_H, wsave_H);
      int n_wsave_W = 4 * n_input_W + 15;
      double *wsave_W = new double[n_wsave_W];
      cffti_( &n_input_W, wsave_W);

      std::complex<double> *col_tmp = new std::complex<double>[n_input_H];

      // Apply 1D FFT to each column of the 2D array (array(id_row,id_column))
      for(int j=0; j<n_input_W; ++j)
      {
        // Copy the column into the C array
        for( int i=0; i<n_input_H; ++i)
          col_tmp[i] = A(i,j);

        // Compute the FFT of one column
        cfftf_( &n_input_H, reinterpret_cast<complex_*>(col_tmp), wsave_H);

        // Update the column
        for( int i=0; i<n_input_H; ++i)
          res(i,j) = col_tmp[i];
      }

      // Apply 1D FFT to each row of the resulting matrix
      for(int i=0; i<n_input_H; ++i)
      {
        // Compute the FFT of one row
        cfftf_( &n_input_W, reinterpret_cast<complex_*>(&(res.data()[i*n_input_W])), wsave_W);
      }

      // Deallocate memory
      delete [] col_tmp;
      delete [] wsave_H;
      delete [] wsave_W;

      return res;
    }


    /**
     * @brief Compute the 2D inverse FFT of a 2D blitz array
     */
    blitz::Array<std::complex<double>,2> 
      ifft(const blitz::Array<std::complex<double>,2>& A)
    {
      // Make a copy of the blitz array. This will allocate a new memory area
      // which will be used to stored the result of the FFT.
      blitz::Array<std::complex<double>,2> res(A.extent(0), A.extent(1));

      // Declare variables for Lapack
      int n_input_H = res.extent(0);
      int n_input_W = res.extent(1);
      // According to Lapack documentation, the work array must be 
      // dimensioned at least 4*n+15, n being the size of the signal to 
      // process.
      int n_wsave_H = 4 * n_input_H + 15;
      double *wsave_H = new double[n_wsave_H];
      cffti_( &n_input_H, wsave_H);
      int n_wsave_W = 4 * n_input_W + 15;
      double *wsave_W = new double[n_wsave_W];
      cffti_( &n_input_W, wsave_W);

      std::complex<double> *col_tmp = new std::complex<double>[n_input_H];

      // Apply 1D inverse FFT to each column of the 2D array (array(id_row,id_column))
      for(int j=0; j<n_input_W; ++j)
      {
        // Copy the column into the C array
        for( int i=0; i<n_input_H; ++i)
          col_tmp[i] = A(i,j);

        // Compute the inverse FFT of one column
        cfftb_( &n_input_H, reinterpret_cast<complex_*>(col_tmp), wsave_H);

        // Update the column
        for( int i=0; i<n_input_H; ++i)
          res(i,j) = col_tmp[i];
      }

      // Apply 1D FFT to each row of the resulting matrix
      for(int i=0; i<n_input_H; ++i)
      {
        // Compute the FFT of one row
        cfftb_( &n_input_W, reinterpret_cast<complex_*>(&(res.data()[i*n_input_W])), wsave_W);
      }

      // Deallocate memory
      delete [] col_tmp;
      delete [] wsave_H;
      delete [] wsave_W;

      // Rescale the result by the size of the input 
      // (as this is not performed by Lapack)
      int n_input = n_input_H * n_input_W;
      res /= static_cast<double>(n_input);

      return res;
    }

  }
}

