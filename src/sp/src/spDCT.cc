#include "sp/spDCT.h"

namespace Torch {

  spDCT::spDCT(bool inverse_)
    :	spCore(), inverse(inverse_), sig(0), cos_coef1(0), cos_coef2(0)
  {
    addBOption("verbose", false, "verbose");
  }


  spDCT::~spDCT()
  {
    if(sig != 0) delete sig;
    if(cos_coef1 != 0) delete cos_coef1;
    if(cos_coef2 != 0) delete cos_coef2;
  }

  bool spDCT::checkInput(const Tensor& input) const
  {
    if (input.nDimension() == 1 || input.nDimension() == 2) {
      return true;
    }
    else {
      warning("spDCT(): incorrect number of dimensions for the input \
        tensor.");
      return false;
    }

    // OK
    return true;
  }


  bool spDCT::allocateOutput(const Tensor& input)
  {
    bool verbose = getBOption("verbose");

    if (	m_output == 0 )
    {
      cleanup();

      if(sig != 0) delete sig;
      if(cos_coef1 != 0) delete cos_coef1;
      if(cos_coef2 != 0) delete cos_coef2;

      sig = cos_coef1 = cos_coef2 = 0;


      if (input.nDimension() == 1)
      {
        if(verbose) print("spDCT::allocateOutput() DCT 1D ...\n");

        N = input.size(0);

        m_n_outputs = 1;
        m_output = new Tensor*[m_n_outputs];
        m_output[0] = new FloatTensor(N);

        sig = new DoubleTensor(N);
        cos_coef1 = new DoubleTensor(2*N+1);
      }
      else if (input.nDimension() == 2)
      {
        if(verbose) print("spDCT::allocateOutput() DCT 2D ...\n");

        H = input.size(0);
        W = input.size(1);
        N = H*W;

        m_n_outputs = 1;
        m_output = new Tensor*[m_n_outputs];
        m_output[0] = new FloatTensor(H,W);

        sig = new DoubleTensor(H,W);
        cos_coef1 = new DoubleTensor(2*H+1);
        cos_coef2 = new DoubleTensor(2*W+1);
      }
    }

    return true;
  }


  bool spDCT::initCosArray(const int N)
  {
    int twoN = 2*N;
    if( cos_coef1 == 0 || cos_coef1->size(0) != twoN+1) {
      warning("spDCT::initCosArray(): Array for cosine coefficients \
        was not allocated correctly.");
      return false;
    }

    for(int i=0; i <= twoN; ++i)
      (*cos_coef1)(i) = cos( M_PI * i / twoN);

    return true;
  }


  bool spDCT::initCosArray(const int H, const int W)
  {
    int twoH = 2*H;
    int twoW = 2*W;
    if( cos_coef1 == 0 || cos_coef1->size(0) != twoH+1 ||
        cos_coef2 == 0 || cos_coef2->size(0) != twoW+1 ) 
    {
      warning("spDCT::initCosArray(): Arrays for cosine coefficients \
        were not allocated correctly.");
      return false;
    }

    for(int i=0; i <= twoH; ++i)
      (*cos_coef1)(i) = cos( M_PI * i / twoH);

    for(int i=0; i <= twoW; ++i)
      (*cos_coef2)(i) = cos( M_PI * i / twoW);

    return true;
  }


  bool spDCT::processInput(const Tensor& input)
  {
    if (input.nDimension() == 1)
    {
      sig->copy(&input);

      if(inverse)
      {
        // Initialize the array of cosine values
        if( !initCosArray(N) )
          return false;

        // Allocate and initialize double precision working array
        double *w=new double[N];
        for(int k=0; k<N; ++k) w[k] = 0.;

        // Preprocess the signal with the multiplicative factors
        double sqrt1N = sqrt(1./N);
        double sqrt2N = sqrt(2./N);
        for(int k=0; k<N; ++k)
          (*sig)(k) = (*sig)(k) * (k==0?sqrt1N:sqrt2N);

        // Compute the inverse DCT
        for(int k=0; k<N; ++k)
          for(int n=0; n<N; ++n)
          {
            // Force the modulus values to be in [0,4*N[ 
            //   (values returned by operator % might be negative)
            int idx = ( ((2*k+1)*n % (4*N)) + 4*N ) % (4*N);

            // Use the symmetry of the cosine function over half a period
            if( idx > 2*N) idx = 4*N - idx;

            // Update the coefficient: 
            // cos(PI*idx/(2*N)) <-> cos(PI*(2*k+1)*n/(2*N))
            w[k] += (*sig)(n) * (*cos_coef1)(idx);
          }

        // Update the output tensor
        FloatTensor *F = (FloatTensor *) m_output[0];
        for(int k=0; k < N; ++k) (*F)(k) = static_cast<float>(w[k]);
      }
      else
      {
        // Initialize the array of cosine values
        if( !initCosArray(N) )
          return false;

        // Allocate and initialize double precision working array
        double *w=new double[N];
        for(int k=0; k<N; ++k) w[k] = 0.;

        // Compute the DCT
        for(int k=0; k<N; ++k)
          for(int n=0; n<N; ++n)
          {
            // Force the modulus values to be in [0,4*N[ 
            //   (values returned by operator % might be negative)
            int idx = ( ((2*n+1)*k % (4*N)) + 4*N ) % (4*N);

            // Use the symmetry of the cosine function over half a period
            if( idx > 2*N) idx = 4*N - idx;

            // Update the coefficient: 
            // cos(PI*idx/(2*N)) <-> cos(PI*(2*n+1)*k/(2*N))
            w[k] += (*sig)(n) * (*cos_coef1)(idx);
          }

        // Update the output tensor with the scaled values
        double sqrt1N = sqrt(1./N);
        double sqrt2N = sqrt(2./N);
        FloatTensor *F = (FloatTensor *) m_output[0];
        for(int k=0; k < N; ++k) 
          (*F)(k) = static_cast<float>(w[k]*(k==0?sqrt1N:sqrt2N));
      }
    }
    else if (input.nDimension() == 2)
    {
      sig->copy(&input);

      if(inverse)
      {
        // Initialize the two arrays of cosine values
        if( !initCosArray(H,W) )
          return false;

        // Allocate and initialize the double precision working array
        double *w=new double[N];
        for(int k=0; k<N; ++k) w[k] =0.;

        // Preprocess the signal with the multiplicative factors
        double sqrt1H = sqrt(1./H);
        double sqrt2H = sqrt(2./H);
        double sqrt1W = sqrt(1./W);
        double sqrt2W = sqrt(2./W);
        for(int k_h=0; k_h<H; ++k_h)
          for(int k_w=0; k_w<W; ++k_w)
            (*sig)(k_h,k_w) = (*sig)(k_h,k_w) * 
              (k_h==0?sqrt1H:sqrt2H) * (k_w==0?sqrt1W:sqrt2W);

        // Compute the DCT
        for(int k_h=0; k_h<H; ++k_h)
          for(int k_w=0; k_w<W; ++k_w)
            for(int n_h=0; n_h<H; ++n_h)
              for(int n_w=0; n_w<W; ++n_w)
              {
                // Force the modulus values to be in [0,4*H[ 
                //   (values returned by operator % might be negative)
                int idh = ( ((2*k_h+1)*n_h % (4*H)) + 4*H ) % (4*H);

                // Use the symmetry of the cosine function over half a period
                if( idh > 2*H) idh = 4*H - idh;

                // Force the modulus values to be in [0,4*W[ 
                //   (values returned by operator % might be negative)
                int idw = ( ((2*k_w+1)*n_w % (4*W)) + 4*W ) % (4*W);

                // Use the symmetry of the cosine function over half a period
                if( idw > 2*W) idw = 4*W - idw;

                // Update the coefficient: 
                // cos(PI*idx/(2*X)) <-> cos(PI*(2*n_x+1)*k_x/(2*N))
                w[k_h*W+k_w] += (*sig)(n_h,n_w) * 
                  (*cos_coef1)(idh) * (*cos_coef2)(idw);
              }

        // Update the output tensor
        FloatTensor *F = (FloatTensor *) m_output[0];
        for(int k_h=0; k_h < H; ++k_h)
          for(int k_w=0; k_w < W; ++k_w)
            (*F)(k_h,k_w) = static_cast<float>(w[k_h*W+k_w]);
      }
      else
      {
        // Initialize the two arrays of cosine values
        if( !initCosArray(H,W) )
          return false;

        // Allocate and initialize double precision working array
        double *w=new double[N];
        for(int k=0; k<N; ++k) w[k] = 0.;

        // Compute the DCT
        for(int k_h=0; k_h<H; ++k_h)
          for(int k_w=0; k_w<W; ++k_w)
            for(int n_h=0; n_h<H; ++n_h)
              for(int n_w=0; n_w<W; ++n_w)
              {
                // Force the modulus values to be in [0,4*H[ 
                //   (values returned by operator % might be negative)
                int idh = ( ((2*n_h+1)*k_h % (4*H)) + 4*H ) % (4*H);

                // Use the symmetry of the cosine function over half a period
                if( idh > 2*H) idh = 4*H - idh;

                // Force the modulus values to be in [0,4*W[ 
                //   (values returned by operator % might be negative)
                int idw = ( ((2*n_w+1)*k_w % (4*W)) + 4*W ) % (4*W);

                // Use the symmetry of the cosine function over half a period
                if( idw > 2*W) idw = 4*W - idw;

                // Update the coefficient: 
                // cos(PI*idx/(2*X)) <-> cos(PI*(2*n_x+1)*k_x/(2*N))
                w[k_h*W+k_w] += (*sig)(n_h,n_w) * 
                  (*cos_coef1)(idh) * (*cos_coef2)(idw);
              }

        // Update the output tensor
        double sqrt1H = sqrt(1./H);
        double sqrt2H = sqrt(2./H);
        double sqrt1W = sqrt(1./W);
        double sqrt2W = sqrt(2./W);
        FloatTensor *F = (FloatTensor *) m_output[0];
        for(int k_h=0; k_h < H; ++k_h)
          for(int k_w=0; k_w < W; ++k_w)
            (*F)(k_h,k_w) = static_cast<float>(w[k_h*W+k_w]*
              (k_h==0?sqrt1H:sqrt2H)*(k_w==0?sqrt1W:sqrt2W));
      }
    }

    // OK
    return true;
  }

}

