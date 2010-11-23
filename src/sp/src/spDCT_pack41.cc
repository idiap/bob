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

    if (input.nDimension() == 1 || input.nDimension() == 2)
    {
      // OK
      return true;
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
        m_output[0] = new FloatTensor(N);

        R = new DoubleTensor(N);
      }
      else if (input.nDimension() == 2)
      {
        if(verbose) print("spDCT_pack41::allocateOutput() DCT 2D ...\n");

        H = input.size(0);
        W = input.size(1);

        m_n_outputs = 1;
        m_output = new Tensor*[m_n_outputs];
        m_output[0] = new FloatTensor(H,W);

        R = new DoubleTensor(H,W);
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
        FloatTensor *F = (FloatTensor *) m_output[0];
        for(int i=0; i < N; ++i) (*F)(i) = static_cast<float>(x[i]);

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
        FloatTensor *F = (FloatTensor *) m_output[0];
        double sqrt2N = sqrt(2./N);
        for(int i=0; i < N; ++i) (*F)(i) = static_cast<float>(x[i]/4.*(i==0?sqrt(1./N):sqrt2N));

        // Deallocate memory
        delete [] wsave;
        delete [] x;
      }
    }
    else if (input.nDimension() == 2)
    {
      R->copy(&input);

      if( inverse)
      {
        // Precompute multiplicative factors
        double sqrt1H=sqrt(1./H);
        double sqrt2H=sqrt(2./H);
        double sqrt1W=sqrt(1./W);
        double sqrt2W=sqrt(2./W);

        // Declare variables
        int n_wsave_W=3*W+15;
        double *wsave_W = new double[n_wsave_W];
        cosqi_( &W, wsave_W);
        int n_wsave_H=3*H+15;
        double *wsave_H = new double[n_wsave_H];
        cosqi_( &H, wsave_H);

        // Allocate memory for C/Fortran arrays
        double *full_x = new double[H*W];
        double *x = new double[W];

        // Apply 1D DCT to each row of the tensor (tensor_x(id_row,id_column))
        for(int j=0; j<H; ++j)
        {
          // Initialize C/Fortran array for FFTPACK
          // Copy the double tensor into the C/Fortran array
          for(int i=0; i < W; ++i) {
            x[i] = (*R)(j,i)*16./(i==0?sqrt1W:sqrt2W)/(j==0?sqrt1H:sqrt2H);
          }

          // Compute the DCT of one row
          cosqf_( &W, x, wsave_W);

          // Copy resulting values into the large C/Fortran array
          for(int i=0; i < W; ++i)
            full_x[j+i*H] = x[i];
        }

        // Apply 1D DCT to each column of the resulting matrix
        for(int i=0; i < W; ++i)
        {
          // Compute the DCT of one column
          cosqf_( &H, full_x+i*H, wsave_H);
        }

        // Update the output tensor with the computed values
        FloatTensor *F = (FloatTensor *) m_output[0];
        for(int i=0; i < W; ++i) {
          int iH = i*H;
          for(int j=0; j < H; ++j)
            (*F)(j,i) = static_cast<float>(full_x[j+iH]/(16.*W*H));
        }

        // Deallocate memory
        delete [] wsave_W;
        delete [] wsave_H;
        delete [] full_x;
        delete [] x;
      }
      else
      {
        // Precompute multiplicative factors
        double sqrt1H=sqrt(1./H);
        double sqrt2H=sqrt(2./H);
        double sqrt1W=sqrt(1./W);
        double sqrt2W=sqrt(2./W);

        // Declare variables
        int n_wsave_W=3*W+15;
        double *wsave_W = new double[n_wsave_W];
        cosqi_( &W, wsave_W);
        int n_wsave_H=3*H+15;
        double *wsave_H = new double[n_wsave_H];
        cosqi_( &H, wsave_H);

        // Allocate memory for C/Fortran arrays
        double *full_x = new double[H*W];
        double *x = new double[W];

        // Apply 1D DCT to each row of the tensor (tensor_x(id_row,id_column))
        for(int j=0; j<H; ++j)
        {
          // Initialize C/Fortran array for FFTPACK
          // Copy the double tensor into the C/Fortran array
          for(int i=0; i < W; ++i) {
            x[i] = (*R)(j,i);
          }

          // Compute the DCT of one row
          cosqb_( &W, x, wsave_W);

          // Copy resulting values into the large C/Fortran array
          for(int i=0; i < W; ++i)
            //full_x[j+i*H] = x[i]/4.*(i==0?sqrt1W:sqrt2W);
            full_x[j+i*H] = x[i];
        }

        // Apply 1D DCT to each column of the resulting matrix
        for(int i=0; i < W; ++i)
        {
          // Compute the DCT of one column
          cosqb_( &H, full_x+i*H, wsave_H);
        }

        // Update the output tensor with the computed values
        FloatTensor *F = (FloatTensor *) m_output[0];
        for(int i=0; i < W; ++i) {
          int iH = i*H;
          for(int j=0; j < H; ++j)
            //(*F)(j,i) = static_cast<float>(full_x[j+iH]/4.*(j==0?sqrt1H:sqrt2H));
            (*F)(j,i) = static_cast<float>(full_x[j+iH]/16.*(j==0?sqrt1H:sqrt2H)*(i==0?sqrt1W:sqrt2W));
        }

        // Deallocate memory
        delete [] wsave_W;
        delete [] wsave_H;
        delete [] full_x;
        delete [] x;
      }
    }

    // OK
    return true;
  }

  /////////////////////////////////////////////////////////////////////////
}

