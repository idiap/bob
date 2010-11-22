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

    R = new DoubleTensor;
    I = new DoubleTensor;
  }

  /////////////////////////////////////////////////////////////////////////
  // Destructor
  spFFT_pack41::~spFFT_pack41()
  {
    delete I;
    delete R;
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
        warning("spFFT_pack41(): FFT2D not yet ready");
        return false;
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
        warning("spFFT_pack41(): FFT2D not yet ready");
        return false;
      }

      warning("spFFT_pack41(): FFT2D not yet ready");
      return false;
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
        m_output[0] = new DoubleTensor(N, 2);
      }
      else if (input.nDimension() == 2)
      {
        if(inverse)
        {
          //print("spFFT_pack41::allocateOutput() assuming inverse FFT 1D ...\n");

          N = input.size(0);

          m_n_outputs = 1;
          m_output = new Tensor*[m_n_outputs];
          m_output[0] = new DoubleTensor(N);
        }
        else
        {
        /* Not yet implemented 
           if(verbose) print("spDCT_pack41::allocateOutput() DCT 2D ...\n");
          //print("spFFT_pack41::allocateOutput() assuming FFT 2D ...\n");

          H = input.size(0);
          W = input.size(1);

          m_n_outputs = 1;
          m_output = new Tensor*[m_n_outputs];
          m_output[0] = new FloatTensor(H,W,2);*/
        }
      }
      else if (input.nDimension() == 3)
      {
        /* Not yet implemented
        //print("spFFT_pack41::allocateOutput() assuming inverse FFT 2D ...\n");

        H = input.size(0);
        W = input.size(1);

        m_n_outputs = 1;
        m_output = new Tensor*[m_n_outputs];
        m_output[0] = new FloatTensor(H,W);*/
      }
    }

    return true;
  }

  /////////////////////////////////////////////////////////////////////////
  // Process some input tensor (the input is checked, the outputs are allocated)
  bool spFFT_pack41::processInput(const Tensor& input)
  {
//  const DoubleTensor* t_input = (DoubleTensor*)&input;

    if (input.nDimension() == 1)
    {
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

      //
      DoubleTensor *F = (DoubleTensor *) m_output[0];
      for(int i=0; i < N; ++i)
      {
        (*F)(i,0) = x[i].real;
        (*F)(i,1) = x[i].imag;
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

        //
        DoubleTensor *F = (DoubleTensor *) m_output[0];
        for(int i=0; i < N; ++i)
          (*F)(i) = x[i].real/N;

        // Deallocate memory
        delete RI;
        delete [] wsave;
        delete [] x;
      }
      else
      {
        /* Not yet implemented */
        ;
      }
    }
    else if (input.nDimension() == 3)
    {
      /* Not yet implemented */
      ;
    }

    // OK
    return true;
  }

  /////////////////////////////////////////////////////////////////////////

}

