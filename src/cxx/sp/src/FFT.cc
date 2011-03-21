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
    void fft(const blitz::Array<std::complex<double>,1>& A,
      blitz::Array<std::complex<double>,1>& B)
    {
      // Check and reindex if required
      if( B.base(0) != 0 ) {
        const blitz::TinyVector<int,1> zero_base = 0;
        B.reindexSelf( zero_base );
      }
      // Check and resize dst if required
      if( B.extent(0) != A.extent(0) )
        B.resize( A.extent(0) );

      // Initialize a blitz array for the result
      blitz::Array<std::complex<double>,1> B_int;

      // Check if B can be directly used
      bool B_ok = checkSafedata(B);

      // Make a reference to B if this is posible.
      // and copy content from A required by Lapack
      if( B_ok )
      {
        B_int.reference( B );
        blitz::Range r_B_int( B_int.lbound(0), B_int.ubound(0) );
        blitz::Range r_A( A.lbound(0), A.ubound(0) );
        B_int(r_B_int) = A(r_A);
      }
      // Otherwise make a full copy
      else
        B_int.reference( blitz::copySafedata(A) );

      // Declare variables for Lapack
      int n_input = B_int.extent(0);
      // According to Lapack documentation, the work array must be 
      // dimensioned at least 4*n+15, n being the size of the signal to 
      // process.
      int n_wsave = 4 * n_input + 15;
      double *wsave = new double[n_wsave];
      // Initialize the work array (with exp coefficients, etc.)
      cffti_( &n_input, wsave);

      // Compute the FFT
      cfftf_( &n_input, reinterpret_cast<complex_*>(B_int.data()), wsave);

      // Deallocate work array
      delete [] wsave;

      // If required, copy the result back to B
      if( !B_ok )
      {
        blitz::Range r_B_int( B_int.lbound(0), B_int.ubound(0) );
        blitz::Range r_B( B.lbound(0), B.ubound(0) );
        B(r_B) = B_int(r_B_int);
      }
    }

    /**
     * @brief Compute the 1D inverse FFT of a 1D blitz array
     */
    void ifft(const blitz::Array<std::complex<double>,1>& A,
      blitz::Array<std::complex<double>,1>& B)
    {
      // Check and reindex if required
      if( B.base(0) != 0 ) {
        const blitz::TinyVector<int,1> zero_base = 0;
        B.reindexSelf( zero_base );
      }
      // Check and resize dst if required
      if( B.extent(0) != A.extent(0) )
        B.resize( A.extent(0) );

      // Initialize a blitz array for the result
      blitz::Array<std::complex<double>,1> B_int;

      // Check if B can be directly used
      bool B_ok = checkSafedata(B);

      // Make a reference to B if this is posible.
      // and copy content from A required by Lapack
      if( B_ok )
      {
        B_int.reference( B );
        blitz::Range r_B_int( B_int.lbound(0), B_int.ubound(0) );
        blitz::Range r_A( A.lbound(0), A.ubound(0) );
        B_int(r_B_int) = A(r_A);
      }
      // Otherwise make a full copy
      else
        B_int.reference( blitz::copySafedata(A) );

      // Declare variables for Lapack
      int n_input = B_int.extent(0);
      // According to Lapack documentation, the work array must be 
      // dimensioned at least 4*n+15, n being the size of the signal to 
      // process.
      int n_wsave = 4 * n_input + 15;
      double *wsave = new double[n_wsave];
      // Initialize the work array (with exp coefficients, etc.)
      cffti_( &n_input, wsave);

      // Compute the FFT
      cfftb_( &n_input, reinterpret_cast<complex_*>(B_int.data()), wsave);

      // Deallocate work array
      delete [] wsave;

      // Rescale the result by the size of the input 
      // (as this is not performed by Lapack)
      B_int /= static_cast<double>(n_input);

      // If required, copy the result back to B
      if( !B_ok )
      {
        blitz::Range r_B_int( B_int.lbound(0), B_int.ubound(0) );
        blitz::Range r_B( B.lbound(0), B.ubound(0) );
        B(r_B) = B_int(r_B_int);
      }
    }


    /**
     * @brief Compute the 2D FFT of a 2D blitz array
     */
    void fft(const blitz::Array<std::complex<double>,2>& A,
      blitz::Array<std::complex<double>,2>& B)
    {
      // Check and reindex if required
      if( B.base(0) != 0 || B.base(1) != 0 ) {
        const blitz::TinyVector<int,2> zero_base = 0;
        B.reindexSelf( zero_base );
      }
      // Check and resize dst if required
      if( B.extent(0) != A.extent(0) || B.extent(1) != A.extent(1) )
        B.resize( A.extent(0), A.extent(1) );

      // Initialize a blitz array for the result
      blitz::Array<std::complex<double>,2> B_int;

      // Check if B can be directly used
      bool B_ok = checkSafedata(B);

      // Make a reference to B if this is posible.
      if( B_ok )
        B_int.reference( B );
      // Otherwise, allocate a new "safe data()" array
      else
        B_int.resize( A.extent(0), A.extent(1) );

      // Declare variables for Lapack
      int n_input_H = B_int.extent(0);
      int n_input_W = B_int.extent(1);
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
          B(i,j) = col_tmp[i];
      }

      // Apply 1D FFT to each row of the resulting matrix
      for(int i=0; i<n_input_H; ++i)
      {
        // Compute the FFT of one row
        cfftf_( &n_input_W, reinterpret_cast<complex_*>(&(B_int.data()[i*n_input_W])), wsave_W);
      }

      // Deallocate memory
      delete [] col_tmp;
      delete [] wsave_H;
      delete [] wsave_W;

      // If required, copy the result back to B
      if( B_ok )
      {
        blitz::Range  r_B_int0( B_int.lbound(0), B_int.ubound(0) ),
                      r_B_int1( B_int.lbound(1), B_int.ubound(1) ),
                      r_B0( B.lbound(0), B.ubound(0) ),
                      r_B1( B.lbound(1), B.ubound(1) );
        B(r_B0, r_B1) = B_int(r_B_int0,r_B_int1);
      }
    }


    /**
     * @brief Compute the 2D inverse FFT of a 2D blitz array
     */
    void ifft(const blitz::Array<std::complex<double>,2>& A,
      blitz::Array<std::complex<double>,2>& B)
    {
      // Check and reindex if required
      if( B.base(0) != 0 || B.base(1) != 0 ) {
        const blitz::TinyVector<int,2> zero_base = 0;
        B.reindexSelf( zero_base );
      }
      // Check and resize dst if required
      if( B.extent(0) != A.extent(0) || B.extent(1) != A.extent(1) )
        B.resize( A.extent(0), A.extent(1) );

      // Initialize a blitz array for the result
      blitz::Array<std::complex<double>,2> B_int;

      // Check if B can be directly used
      bool B_ok = checkSafedata(B);

      // Make a reference to B if this is posible.
      if( B_ok )
        B_int.reference( B );
      // Otherwise, allocate a new "safe data()" array
      else
        B_int.resize( A.extent(0), A.extent(1) );

      // Declare variables for Lapack
      int n_input_H = B_int.extent(0);
      int n_input_W = B_int.extent(1);
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
          B_int(i,j) = col_tmp[i];
      }

      // Apply 1D FFT to each row of the resulting matrix
      for(int i=0; i<n_input_H; ++i)
      {
        // Compute the FFT of one row
        cfftb_( &n_input_W, reinterpret_cast<complex_*>(&(B_int.data()[i*n_input_W])), wsave_W);
      }

      // Deallocate memory
      delete [] col_tmp;
      delete [] wsave_H;
      delete [] wsave_W;

      // Rescale the result by the size of the input 
      // (as this is not performed by Lapack)
      int n_input = n_input_H * n_input_W;
      B_int /= static_cast<double>(n_input);

      // If required, copy the result back to B
      if( B_ok )
      {
        blitz::Range  r_B_int0( B_int.lbound(0), B_int.ubound(0) ),
                      r_B_int1( B_int.lbound(1), B_int.ubound(1) ),
                      r_B0( B.lbound(0), B.ubound(0) ),
                      r_B1( B.lbound(1), B.ubound(1) );
        B(r_B0, r_B1) = B_int(r_B_int0,r_B_int1);
      }
    }

  }
}

