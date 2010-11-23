#include "sp/spDFT.h"

namespace Torch {

  spDFT::spDFT(bool inverse_)
    :	spCore(), inverse(inverse_), sig(0), exp_coef1(0), exp_coef2(0)
  {
  }


  spDFT::~spDFT()
  {
    if(sig != 0) delete sig;
    if(exp_coef1 != 0) delete exp_coef1;
    if(exp_coef2 != 0) delete exp_coef2;
  }


  bool spDFT::checkInput(const Tensor& input) const
  {
    /*
       input	output
       forward	1D 	2D
       inverse	2D	1D
       forward	2D	3D
       inverse	3D	2D
    */
    if (input.nDimension() == 1)
    {
      //print("spDFT::checkInput() assuming DFT 1D ...\n");

      if(inverse) {
        warning("spDFT(): impossible to handle inverse mode with 1D input \
          tensor.");
        return false;
      }
    }
    if (input.nDimension() == 2)
    {
      if(inverse) {
        //print("spDFT::checkInput() assuming inverse DFT 1D ...\n");
        if(input.size(1) != 2) {
          warning("spDFT(): size(1) is not 2 (necessary to handle real and \
            imag parts).");
          return false;
        }
      }
      else {
        //print("spDFT::checkInput() assuming DFT 2D ...\n");
      }
    }
    if (input.nDimension() == 3)
    {
      if(inverse == false) {
        warning("spDFT(): impossible to handle forward mode with 3D input \
          tensor.");
        return false;
      }
      //print("spDFT::checkInput() assuming inverse DFT 2D ...\n");

      if(input.size(2) != 2) {
        warning("spDFT(): size(2) is not 2 (necessary to handle real and \
          imag parts).");
        return false;
      }
    }

    // OK
    return true;
  }


  bool spDFT::allocateOutput(const Tensor& input)
  {
    if (	m_output == 0 )
    {
      // Cleanup inclusive dynamically allocated memory
      cleanup();

  		if(sig != 0) delete sig;
      if(exp_coef1 != 0) delete exp_coef1;
      if(exp_coef2 != 0) delete exp_coef2;

      sig = exp_coef1 = exp_coef2 = 0;


      if (input.nDimension() == 1)
      {
        //print("spDFT::allocateOutput() assuming DFT 1D ...\n");

        N = input.size(0);

        m_n_outputs = 1;
        m_output = new Tensor*[m_n_outputs];
        m_output[0] = new FloatTensor(N, 2);

        sig = new DoubleTensor(N);
        exp_coef1 = new DoubleTensor(N,2);
      }
      else if (input.nDimension() == 2)
      {
        if(inverse)
        {
          //print("spDFT::allocateOutput() assuming inverse DFT 1D ...\n");

          N = input.size(0);

          m_n_outputs = 1;
          m_output = new Tensor*[m_n_outputs];
          m_output[0] = new FloatTensor(N);

          sig = new DoubleTensor(N,2);
          exp_coef1 = new DoubleTensor(N,2);
        }
        else
        {
          //print("spDFT::allocateOutput() assuming DFT 2D ...\n");

          H = input.size(0);
          W = input.size(1);
          N = H*W;

          m_n_outputs = 1;
          m_output = new Tensor*[m_n_outputs];
          m_output[0] = new FloatTensor(H,W,2);

          sig = new DoubleTensor(H,W);
          exp_coef1 = new DoubleTensor(H,2);
          exp_coef2 = new DoubleTensor(W,2);
        }
      }
      else if (input.nDimension() == 3)
      {
        //print("spDFT::allocateOutput() assuming inverse DFT 2D ...\n");

        H = input.size(0);
        W = input.size(1);
        N=H*W;

        m_n_outputs = 1;
        m_output = new Tensor*[m_n_outputs];
        m_output[0] = new FloatTensor(H,W);

        sig = new DoubleTensor(H,W,2);
        exp_coef1 = new DoubleTensor(H,2);
        exp_coef2 = new DoubleTensor(W,2);
      }
    }

    return true;
  }


  bool spDFT::initExpArray(const int NN)
  {
    if( exp_coef1 == 0 || exp_coef1->size(0) != NN)
    {
      warning("spDFT::initExpArray(): Arrays for exponential coefficients \
        were not allocated correctly.");
      return false;
    }

    // Initialize complex exponentials used for the DFT computation
    for(int i=0; i < NN; ++i) {
      (*exp_coef1)(i,0) = cos( -2 * M_PI * i / NN);
      (*exp_coef1)(i,1) = sin( -2 * M_PI * i / NN);
    }

    return true;
  }


  bool spDFT::initExpArray(const int HH, const int WW)
  {
    if( exp_coef1 == 0 || exp_coef1->size(0) != HH ||
        exp_coef2 == 0 || exp_coef2->size(0) != WW)
    {
      warning("spDFT::initExpArray(): Arrays for exponential coefficients \
        were not allocated correctly.");
      return false;
    }

    // Initialize complex exponentials used for the 2D DFT computation
    for(int i=0; i < HH; ++i) {
      (*exp_coef1)(i,0) = cos( -2 * M_PI * i / HH);
      (*exp_coef1)(i,1) = sin( -2 * M_PI * i / HH);
    }

    for(int i=0; i < WW; ++i) {
      (*exp_coef2)(i,0) = cos( -2 * M_PI * i / WW);
      (*exp_coef2)(i,1) = sin( -2 * M_PI * i / WW);
    }

    return true;
  }


  bool spDFT::processInput(const Tensor& input)
  {
    // Direct 1D DFT
    if (input.nDimension() == 1)
    {
      // Initialize the exponential coefficients
      if( !initExpArray(N) )
        return true;

      // Copy the input tensor of unknown type into the DoubleTensor sig
      sig->copy(&input);

      // Allocate working array for real and imaginary parts
      double *w_r = new double[N];
      double *w_i = new double[N];

      // Initialization
      for(int n=0; n<N; n++)
        w_r[n] = w_i[n] = 0.;

      // Compute the 1D DFT
      for(int k=0; k<N; ++k)
        for(int n=0; n<N; ++n) {
          // Force the modulus value to be in [0,N[ 
          //   (as values returned by operator % might be negative)
          int idx = ( (n*k % N) + N ) % N;

          // Update the coefficients
          //   Output[k] += (*Input)(n) * exp(-2 * M_PI *n*k/N);
          w_r[k] += (*sig)(n) * (*exp_coef1)(idx,0);
          w_i[k] += (*sig)(n) * (*exp_coef1)(idx,1);
        }

      // Update the output tensor
			FloatTensor *F = (FloatTensor *) m_output[0];
			for(int k=0; k < N; ++k) {
        (*F)(k,0) = static_cast<float>(w_r[k]);
        (*F)(k,1) = static_cast<float>(w_i[k]);
      }

      delete [] w_r;
      delete [] w_i;
    }
    else if (input.nDimension() == 2)
    {
      // Inverse 1D DFT
      if(inverse)
      {
        // Copy the input tensor of unknown type into the DoubleTensor sig
        sig->copy(&input);

        // Initialize the exponential coefficients
        if( !initExpArray(N) )
          return true;

        // Allocate working array for real (imaginary part is ignored so far)
        double *w_r = new double[N];
        // double *w_i = new double[N]; // imaginary part ignored
 
        // Initialization
        for(int n=0; n<N; n++) {
          w_r[n] = 0.;
          // w_i[n] = 0.; // imaginary part ignored
        }

        // Compute the inverse DFT
        for(int k=0; k<N; ++k)
          for(int n=0; n<N; ++n) {
            // Force the modulus value to be in [0,N[ 
            //   (as values returned by operator % might be negative)
            int idx = ( ((-n*k) % N) + N ) % N;

            // Update the coefficients
            // Output(k) += (*Input)(n) * exp(-2 * M_PI *(-n*k)/N);
            w_r[k]+= (*sig)(n,0)*(*exp_coef1)(idx,0) - 
              (*sig)(n,1)*(*exp_coef1)(idx,1);
            // w_i[k]+= (*sig)(n,0)*(*exp_coef1)(idx,1) + 
            //   (*sig)(n,1)*(*exp_coef1)(idx,0); // imaginary part ignored
          }

        // Update the output tensor with the scaled real part only
	  		FloatTensor *F = (FloatTensor *) m_output[0];
		  	for(int k=0; k < N; ++k) {
          (*F)(k) = static_cast<float>(w_r[k]/N);
          // (*F)(k,1) = static_cast<float>(w_i[k]); // imaginary part ignored
        }

        delete [] w_r;
        // delete [] w_i; // imaginary part ignored
      }
      // Direct 2D DFT
      else
      {
        // Initialize the exponential coefficients
        if( !initExpArray(H,W) )
          return true;

        // Copy the input tensor of unknown type into the DoubleTensor R
        sig->copy(&input);

        // Allocate working arrays for real and imaginary parts
        double *w_r = new double[N];
        double *w_i = new double[N];

        // Initialization
        for(int n=0; n<N; ++n)
          w_r[n] = w_i[n] = 0.;

        // Compute the DFT
        for(int k_h=0; k_h<H; ++k_h)
          for(int k_w=0; k_w<W; ++k_w)
            for(int n_h=0; n_h<H; ++n_h)
              for(int n_w=0; n_w<W; ++n_w)
              {
                // Force the modulus value to be in [0,H[ 
                //   (as values returned by operator % might be negative)
                int idh = ( (n_h*k_h % H) + H ) % H;

                // Force the modulus value to be in [0,W[ 
                //   (as values returned by operator % might be negative)
                int idw = ( (n_w*k_w % W) + W ) % W;

                // Update the coefficients (real and imaginary parts)
                // Input[k_h*W+k_w] += (*Output)(n_h,n_w) * 
                //   exp(M_PI *(2*n_h+1)*k_h/(2*H)) * 
                //   exp(M_PI *(2*n_w+1)*k_w/(2*W));
                w_r[k_h*W+k_w] += (*sig)(n_h,n_w) *
                  ( (*exp_coef1)(idh,0) * (*exp_coef2)(idw,0) - 
                    (*exp_coef1)(idh,1) * (*exp_coef2)(idw,1) );
                w_i[k_h*W+k_w] += (*sig)(n_h,n_w) * 
                  ( (*exp_coef1)(idh,0) * (*exp_coef2)(idw,1) + 
                    (*exp_coef1)(idh,1) * (*exp_coef2)(idw,0) );
              }

        // Update the output tensor
        FloatTensor *F = (FloatTensor *) m_output[0];
			  for(int k_h=0; k_h < H; ++k_h)
			    for(int k_w=0; k_w < W; ++k_w) {
            (*F)(k_h,k_w,0) = static_cast<float>(w_r[k_h*W+k_w]);
            (*F)(k_h,k_w,1) = static_cast<float>(w_i[k_h*W+k_w]);
        }

        delete [] w_r;
        delete [] w_i;
      }
    }
    // Inverse 2D DFT
    else if (input.nDimension() == 3)
    {
      if(inverse)
      {
        // Copy the input tensor of unknown type into a DoubleTensor sig
        sig->copy(&input);

        // Initialize the exponential coefficients
        if( !initExpArray(H,W) )
          return true;

        // Allocate working array for real part (imaginary part ignored)
        double *w_r = new double[N];
        // double *w_i = new double[N]; // imaginary part ignored

        // Initialization
        for(int n=0; n<N; n++) {
          w_r[n] = 0.;
          // w_i[n] = 0.; // imaginary part ignored
        }

        // Compute the inverse DFT
        for(int k_h=0; k_h<H; ++k_h)
          for(int k_w=0; k_w<W; ++k_w)
            for(int n_h=0; n_h<H; ++n_h)
              for(int n_w=0; n_w<W; ++n_w) {
                // Force the modulus value to be in [0,H[ 
                //   (as values returned by operator % might be negative)
                int idh = ( ((-n_h*k_h) % H) + H ) % H;

                // Force the modulus value to be in [0,W[ 
                //   (as values returned by operator % might be negative)
                int idw = ( ((-n_w*k_w) % W) + W ) % W;

                // Update the coefficients
                //Input(k_h,k_w) += (*Output)(n_h,n_w) * 
                //  exp(M_PI *(2*n_h+1)*k_h/(2*H)) * 
                //  exp(M_PI *(2*n_w+1)*k_w/(2*W));
                w_r[k_h*W+k_w] += (*sig)(n_h,n_w,0) * 
                    ( (*exp_coef1)(idh,0) * (*exp_coef2)(idw,0) - 
                      (*exp_coef1)(idh,1) * (*exp_coef2)(idw,1) ) -
                  (*sig)(n_h,n_w,1) * 
                    ( (*exp_coef1)(idh,0) * (*exp_coef2)(idw,1) + 
                      (*exp_coef1)(idh,1) * (*exp_coef2)(idw,0) );
                //  w_i[k_h*W+k_w] += (*sig)(n_h,n_w,0) * 
                //      ( (*exp_coef1)(idh,0) * (*exp_coef2)(idw,1) + 
                //        (*exp_coef1)(idh,1) * (*exp_coef2)(idw,0) ) +
                //    (*sig)(n_h,n_w,1) * 
                //      ( (*exp_coef1)(idh,0) * (*exp_coef2)(idw,0) - 
                //        (*exp_coef1)(idh,1) * (*exp_coef2)(idw,1) );
              }

        // Update the output tensor with the scaled real part only
	  		FloatTensor *F = (FloatTensor *) m_output[0];
        for(int k_h=0; k_h<H; ++k_h)
          for(int k_w=0; k_w<W; ++k_w) {
            (*F)(k_h,k_w) = static_cast<float>(w_r[k_h*W+k_w]/N);
            // imaginary part is ignored so far
            //(*F)(k_h,k_w,1) = static_cast<float>(w_i[k_h*W+k_w]/N);
        }

        delete [] w_r;
        // delete [] w_i; // imaginary part ignored
      }
    }

    // OK
    return true;
  }

}

