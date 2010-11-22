#include "sp/spDCT_pack41.h"

#define MAX(x,y) ((x) > (y) ? (x) : (y))


// Declaration of FORTRAN functions from FFTPACK4.1
extern "C" void cosqi_( int *n, double *wsave);
extern "C" void cosqf_( int *n, double *x, double *wsave);
extern "C" void cosqb_( int *n, double *x, double *wsave);

namespace Torch {

  /////////////////////////////////////////////////////////////////////////
  // Constructor
  spDCT_pack41::spDCT_pack41(bool inverse_)
    :	spCore()
  {
    inverse = inverse_;
    R = NULL;

    addBOption("verbose", false, "verbose");
  }

  /////////////////////////////////////////////////////////////////////////
  // Destructor
  spDCT_pack41::~spDCT_pack41()
  {
    if(R != NULL) delete R;
  }

  //////////////////////////////////////////////////////////////////////////
  // Check if the input tensor has the right dimensions and type
  bool spDCT_pack41::checkInput(const Tensor& input) const
  {

    if (input.nDimension() == 1)
    {
      //OK
      return true;
    }
    else if (input.nDimension() == 2)
    {
      warning("spDCT_pack41(): DCT2D not yet ready");
      return false;
    }
    else
    {
      warning("spDCT_pack41(): incorrect number of dimensions for the input tensor.");
      return false;
    }

    // OK
    return true;
  }

  /////////////////////////////////////////////////////////////////////////
  // Allocate (if needed) the output tensors given the input tensor dimensions
  bool spDCT_pack41::allocateOutput(const Tensor& input)
  {
    bool verbose = getBOption("verbose");

    if (	m_output == 0 )
    {
      cleanup();

      if(R != NULL) delete R;

      if (input.nDimension() == 1)
      {
        if(verbose) print("spDCT_pack41::allocateOutput() DCT 1D ...\n");

        N = input.size(0);

        m_n_outputs = 1;
        m_output = new Tensor*[m_n_outputs];
        m_output[0] = new DoubleTensor(N);

        R = new DoubleTensor(N);
      }
      else if (input.nDimension() == 2)
      {
        /* Not yet implemented 
           if(verbose) print("spDCT_pack41::allocateOutput() DCT 2D ...\n");

           H = input.size(0);
           W = input.size(1);

           m_n_outputs = 1;
           m_output = new Tensor*[m_n_outputs];
           m_output[0] = new DoubleTensor(H,W);

           R = new DoubleTensor(H,W);*/
      }
    }

    return true;
  }

  /////////////////////////////////////////////////////////////////////////
  // Process some input tensor (the input is checked, the outputs are allocated)
  bool spDCT_pack41::processInput(const Tensor& input)
  {
    if (input.nDimension() == 1)
    {
      R->copy(&input);

      if(inverse)
      {
        // Declare variables
        int n_wsave=3*N+15;
        double *wsave = new double[n_wsave];
        cosqi_( &N, wsave);

        // Allocate a C/Fortran array according to the FFTPACK requirements
        double *x = new double[N];

        // Copy the double tensor in the C/Fortran array and
        // Update the coefficients according to the ones of FFTPACK
        double sqrt2N = sqrt(2.*N);
        for(int i=0; i < N; ++i) x[i] = (*R)(i)/(i==0?sqrt(N):sqrt2N);

        // Compute the inverse DCT
        cosqf_( &N, x, wsave);

        // Update the tensor from the C/Fortran array
        DoubleTensor *F = (DoubleTensor *) m_output[0];
        for(int i=0; i < N; ++i) (*F)(i) = x[i];

        // Deallocate memory
        delete [] wsave;
        delete [] x;
      }
      else
      {
        // Declare variables
        int n_wsave=3*N+15;
        double *wsave = new double[n_wsave];
        cosqi_( &N, wsave);

        // Allocate C/Fortran array according to the FFTPACK requirements
        double *x = new double[N];

        // Copy the double tensor in the C/Fortran array
        for(int i=0; i < N; ++i) x[i] = (*R)(i);

        // Compute the direct DCT
        cosqb_( &N, x, wsave);

        // Update the tensor from the C/Fortran array and
        // scale it with correct coefficients
        DoubleTensor *F = (DoubleTensor *) m_output[0];
        double sqrt2N = sqrt(2./N);
        for(int i=0; i < N; ++i) (*F)(i) = x[i]/4.*(i==0?sqrt(1./N):sqrt2N);

        // Deallocate memory
        delete [] wsave;
        delete [] x;
      }
    }
    else if (input.nDimension() == 2)
    {
      /* Not yet implemented */
      ;
    }

    // OK
    return true;
  }

  /////////////////////////////////////////////////////////////////////////
}

