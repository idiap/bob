#include "sp/spFFT_pack41.h"

#define MAX(x,y) ((x) > (y) ? (x) : (y))


// To deal with Fotran complex type
typedef struct { double real, imag; } complex_;

// Declaration of FORTRAN functions from FFTPACK4.1
extern "C" void cffti_( int *n, double *wsave);
extern "C" void cfftf_( int *n, complex_ *x, double *wsave);
extern "C" void cfftb_( int *n, complex_ *x, double *wsave);


namespace Torch {

  /////////////////////////////////////////////////////////////////////////
  // Constructor
  spFFT_pack41::spFFT_pack41(bool inverse_)
    :	spCore()
  {
    inverse = inverse_;
  }

  /////////////////////////////////////////////////////////////////////////
  // Destructor
  spFFT_pack41::~spFFT_pack41()
  {
  }

  //////////////////////////////////////////////////////////////////////////
  // Check if the input tensor has the right dimensions and type
  bool spFFT_pack41::checkInput(const Tensor& input) const
  {
    // Accept only tensors of Torch::Float
    //if (input.getDatatype() != Tensor::Double) return false;

    /*
       input	output
       forward	1D 	2D
       inverse	2D	1D
       forward	2D	3D
       inverse	3D	2D
    */

    if (input.nDimension() == 1)
    {
      //print("spFFT_pack41::checkInput() assuming FFT 1D ...\n");

      if(inverse)
      {
        warning("spFFT_pack41(): impossible to handle inverse mode with 1D input tensor.");
        return false;
      }
    }

    if (input.nDimension() == 2)
    {
      if(inverse)
      {
        //print("spFFT_pack41::checkInput() assuming inverse FFT 1D ...\n");
        return true;
      }
      else
      {
        //print("spFFT_pack41::checkInput() assuming FFT 2D ...\n");
        return true;
      }
    }

    if (input.nDimension() == 3)
    {
      //print("spFFT_pack41::checkInput() assuming inverse FFT 2D ...\n");

      if(inverse == false)
      {
        warning("spFFT_pack41(): impossible to handle forward mode with 3D input tensor.");
        return false;
      }

      if(input.size(2) != 2)
      {
        warning("spFFT_pack41(): The third dimension should of the 3D input tensor should be equal to 2.");
        return false;
      }

      return true;
    }

    // OK
    return true;
  }

  /////////////////////////////////////////////////////////////////////////
  // Allocate (if needed) the output tensors given the input tensor dimensions
  bool spFFT_pack41::allocateOutput(const Tensor& input)
  {
    if (	m_output == 0 )
    {
      cleanup();

      if (input.nDimension() == 1)
      {
        //print("spFFT_pack41::allocateOutput() assuming FFT 1D ...\n");

        N = input.size(0);

        m_n_outputs = 1;
        m_output = new Tensor*[m_n_outputs];
        m_output[0] = new FloatTensor(N, 2);
      }
      else if (input.nDimension() == 2)
      {
        if(inverse)
        {
          //print("spFFT_pack41::allocateOutput() assuming inverse FFT 1D ...\n");

          N = input.size(0);

          m_n_outputs = 1;
          m_output = new Tensor*[m_n_outputs];
          m_output[0] = new FloatTensor(N);
        }
        else
        {
          //if(verbose) print("spDCT_pack41::allocateOutput() DCT 2D ...\n");
          //print("spFFT_pack41::allocateOutput() assuming FFT 2D ...\n");

          H = input.size(0);
          W = input.size(1);

          m_n_outputs = 1;
          m_output = new Tensor*[m_n_outputs];
          m_output[0] = new FloatTensor(H,W,2);
        }
      }
      else if (input.nDimension() == 3)
      {
        //print("spFFT_pack41::allocateOutput() assuming inverse FFT 2D ...\n");

        H = input.size(0);
        W = input.size(1);

        m_n_outputs = 1;
        m_output = new Tensor*[m_n_outputs];
        m_output[0] = new FloatTensor(H,W);
      }
    }

    return true;
  }

  /////////////////////////////////////////////////////////////////////////
  // Process some input tensor (the input is checked, the outputs are allocated)
  bool spFFT_pack41::processInput(const Tensor& input)
  {
    if (input.nDimension() == 1)
    {
      // Useless copy into RI
      // Can be copied directly into the C/Fortran array
      DoubleTensor *RI = new DoubleTensor(N);

      RI->copy(&input);

      // Declare variables
      int n_wsave=4*N+15;
      double *wsave = new double[n_wsave];
      cffti_( &N, wsave);

      // Initialize C/Fortran array for FFTPACK
      complex_ *x = new complex_[N];

      // Copy the double tensor into the C/Fortran array
      for(int i=0; i < N; ++i) {
        x[i].real = (*RI)(i);
        x[i].imag = 0.0;
      }

      // Compute the FFT
      cfftf_( &N, x, wsave);

      // Update the output tensor
      FloatTensor *F = (FloatTensor *) m_output[0];
      for(int i=0; i < N; ++i)
      {
        (*F)(i,0) = static_cast<float>(x[i].real);
        (*F)(i,1) = static_cast<float>(x[i].imag);
      }

      // Deallocate memory
      delete RI;
      delete [] wsave;
      delete [] x;
    }
    else if (input.nDimension() == 2)
    {
      if(inverse)
      {
        // Useless copy into RI
        // Can be copied directly into the C/Fortran array
        DoubleTensor *RI = new DoubleTensor(N);

        RI->copy(&input);

        // Declare variables
        int n_wsave=4*N+15;
        double *wsave = new double[n_wsave];
        cffti_( &N, wsave);

        // Initialize C/Fortran array for FFTPACK
        complex_ *x = new complex_[N];

        // Copy the double tensor into the C/Fortran array
        for(int i=0; i < N; ++i) {
          x[i].real = (*RI)(i,0);
          x[i].imag = (*RI)(i,1);
       }

        // Compute the FFT
        cfftb_( &N, x, wsave);

        // Update the output tensor (return real part only)
        FloatTensor *F = (FloatTensor *) m_output[0];
        for(int i=0; i < N; ++i)
          (*F)(i) = static_cast<float>(x[i].real/N);

        // Deallocate memory
        delete RI;
        delete [] wsave;
        delete [] x;
      }
      else
      {
        // Useless copy into RI
        // Can be copied directly into the C/Fortran array
        DoubleTensor *RI = new DoubleTensor(H,W);

        RI->copy(&input);

        // Declare variables
        int n_wsave_W=4*W+15;
        double *wsave_W = new double[n_wsave_W];
        cffti_( &W, wsave_W);
        int n_wsave_H=4*H+15;
        double *wsave_H = new double[n_wsave_H];
        cffti_( &H, wsave_H);

        // Allocate memory for C/Fortran arrays
        complex_ *full_x = new complex_[H*W];
        complex_ *x = new complex_[W];

        // Apply 1D FFT to each row of the tensor (tensor_x(id_row,id_column))
        for(int j=0; j<H; ++j)
        {
          // Initialize C/Fortran array for FFTPACK
          // Copy the double tensor into the C/Fortran array
          for(int i=0; i < W; ++i) {
            x[i].real = (*RI)(j,i);
            x[i].imag = 0.0;
          }

          // Compute the FFT of one row
          cfftf_( &W, x, wsave_W);

          // Copy resulting values into the large C/Fortran array
          for(int i=0; i < W; ++i) {
            full_x[j+i*H].real = x[i].real;
            full_x[j+i*H].imag = x[i].imag;
          }
        }

        // Apply 1D FFT to each column of the resulting matrix
        for(int i=0; i < W; ++i)
        {
          // Compute the FFT of one column
          cfftf_( &H, full_x+i*H, wsave_H);
        }

        // Update the output tensor with the computed values
        FloatTensor *F = (FloatTensor *) m_output[0];
        for(int i=0; i < W; ++i)
          for(int j=0; j < H; ++j) {
            int iH = i*H;
            (*F)(j,i,0) = static_cast<float>(full_x[j+iH].real);
            (*F)(j,i,1) = static_cast<float>(full_x[j+iH].imag);
          }

        // Deallocate memory
        delete RI;
        delete [] wsave_W;
        delete [] wsave_H;
        delete [] x;
      }
    }
    else if (input.nDimension() == 3)
    {
      if( inverse)
      {
        /* Not yet implemented */
        // Useless copy into RI
        // Can be copied directly into the C/Fortran array
        DoubleTensor *RI = new DoubleTensor(H,W,2);

        RI->copy(&input);

        // Declare variables
        int n_wsave_W=4*W+15;
        double *wsave_W = new double[n_wsave_W];
        cffti_( &W, wsave_W);
        int n_wsave_H=4*H+15;
        double *wsave_H = new double[n_wsave_H];
        cffti_( &H, wsave_H);

        // Allocate memory for C/Fortran arrays
        N=H*W;
        complex_ *full_x = new complex_[N];
        complex_ *x = new complex_[W];

        // Apply 1D inverse FFT to each row of the tensor (tensor_x(id_row,id_column))
        for(int j=0; j<H; ++j)
        {
          // Initialize C/Fortran array for FFTPACK
          // Copy the double tensor into the C/Fortran array
          for(int i=0; i < W; ++i) {
            x[i].real = (*RI)(j,i,0);
            x[i].imag = (*RI)(j,i,1);
          }

          // Compute the inverse FFT of one row
          cfftb_( &W, x, wsave_W);

          // Copy resulting values into the large C/Fortran array
          for(int i=0; i < W; ++i) {
            full_x[j+i*H].real = x[i].real;
            full_x[j+i*H].imag = x[i].imag;
          }
        }

        // Apply 1D inverse FFT to each column of the resulting matrix
        for(int i=0; i < W; ++i)
        {
          // Compute the inverse FFT of one column
          cfftb_( &H, full_x+i*H, wsave_H);
        }

        // Update the output tensor with the computed values
        FloatTensor *F = (FloatTensor *) m_output[0];
        for(int j=0; j < H; ++j)
          for(int i=0; i < W; ++i) {
            (*F)(j,i) = static_cast<float>(full_x[j+i*H].real/N);
          }

        // Deallocate memory
        delete RI;
        delete [] wsave_W;
        delete [] wsave_H;
        delete [] x;
      }
    }

    // OK
    return true;
  }

  /////////////////////////////////////////////////////////////////////////

}

