#include "sp/spFFT.h"
#include "oourafft/ooura.h"

#define MAX(x,y) ((x) > (y) ? (x) : (y))

namespace Torch {

/////////////////////////////////////////////////////////////////////////
// Constructor

spFFT::spFFT(bool inverse_)
	:	spCore()
{
	inverse = inverse_;
			
	R = new FloatTensor;
	I = new FloatTensor;
}

/////////////////////////////////////////////////////////////////////////
// Destructor

spFFT::~spFFT()
{
	delete I;
	delete R;
}

//////////////////////////////////////////////////////////////////////////
// Check if the input tensor has the right dimensions and type

bool spFFT::checkInput(const Tensor& input) const
{
	// Accept only tensors of Torch::Float
	if (input.getDatatype() != Tensor::Float) return false;


	/*
			input	output
	     forward	1D 	2D
	     inverse	2D	1D
	     forward	2D	3D
	     inverse	3D	2D

	*/

	if (input.nDimension() == 1)
	{
		//print("spFFT::checkInput() assuming FFT 1D ...\n");

	   	if(inverse)
		{
			warning("spFFT(): impossible to handle inverse mode with 1D input tensor.");
			return false;
		}

		int N_ = input.size(0);

		unsigned int nn = nexthigher(N_); 
		
		if(N_ != (int) nn)
		{
			warning("spFFT(): size(0) is not a power of 2.");
			return false;
		}
	}
	
	if (input.nDimension() == 2)
	{
	   	if(inverse)
		{
			//print("spFFT::checkInput() assuming inverse FFT 1D ...\n");

			int N_ = input.size(0);
			unsigned int nn = nexthigher(N_); 
			if(N_ != (int) nn)
			{
				warning("spFFT(): size(0) is not a power of 2.");
				return false;
			}
		}
		else
		{
			//print("spFFT::checkInput() assuming FFT 2D ...\n");

			int N_ = input.size(0);
			unsigned int nn = nexthigher(N_); 
			if(N_ != (int) nn)
			{
				warning("spFFT(): size(0) is not a power of 2.");
				return false;
			}
			N_ = input.size(1);
			nn = nexthigher(N_); 
			if(N_ != (int) nn)
			{
				warning("spFFT(): size(1) is not a power of 2.");
				return false;
			}
		}
	}
	
	if (input.nDimension() == 3)
	{
		//print("spFFT::checkInput() assuming inverse FFT 2D ...\n");

	   	if(inverse == false)
		{
			warning("spFFT(): impossible to handle forward mode with 3D input tensor.");
			return false;
		}

		if(input.size(2) != 2)
		{
			warning("spFFT(): size(2) is not 2 (necessary to handle real and imag parts).");
			return false;
		}

		int N_ = input.size(0);
		unsigned int nn = nexthigher(N_); 
		if(N_ != (int) nn)
		{
			warning("spFFT(): size(0) is not a power of 2.");
			return false;
		}
		N_ = input.size(1);
		nn = nexthigher(N_); 
		if(N_ != (int) nn)
		{
			warning("spFFT(): size(1) is not a power of 2.");
			return false;
		}
	}
	
	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Allocate (if needed) the output tensors given the input tensor dimensions

bool spFFT::allocateOutput(const Tensor& input)
{
	if (	m_output == 0 )
	{
		cleanup();
	
		if (input.nDimension() == 1)
		{
			//print("spFFT::allocateOutput() assuming FFT 1D ...\n");

			N = input.size(0);

			m_n_outputs = 1;
			m_output = new Tensor*[m_n_outputs];
			m_output[0] = new FloatTensor(N, 2);
		}
		else if (input.nDimension() == 2)
		{
		   	if(inverse)
			{
				//print("spFFT::allocateOutput() assuming inverse FFT 1D ...\n");

				N = input.size(0);

				m_n_outputs = 1;
				m_output = new Tensor*[m_n_outputs];
				m_output[0] = new FloatTensor(N);
			}
			else
			{
				//print("spFFT::allocateOutput() assuming FFT 2D ...\n");

				H = input.size(0);
				W = input.size(1);

				m_n_outputs = 1;
				m_output = new Tensor*[m_n_outputs];
				m_output[0] = new FloatTensor(H,W,2);
			}
		}
		else if (input.nDimension() == 3)
		{
			//print("spFFT::allocateOutput() assuming inverse FFT 2D ...\n");

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

bool spFFT::processInput(const Tensor& input)
{
	const FloatTensor* t_input = (FloatTensor*)&input;

	if (input.nDimension() == 1)
	{
		FloatTensor *RI = new FloatTensor(N);

		RI->copy(t_input);

#ifdef HAVE_OOURAFFT
		// Workspace of Ooura FFT
		int *ip;
		double *w;
		double *a;

		// Alloc workspace
		a = alloc_1d_double(2*N);
		ip = alloc_1d_int(2 + (int) sqrt(N + 0.5));
		w = alloc_1d_double(N * 5 / 4);
  
		// Init workspace
		ip[0] = 0; // Possible speed up here as ip[0] == 0 is used to init the rest of ip and the workspace !!
		
		for(int i=0; i < N; i++) {
      a[2*i] = (*RI)(i);
      a[2*i+1] = 0.0;
    }

		// Complex Discrete Fourier Transform routine
		cdft(2*N, -1, a, ip, w);

		//
		FloatTensor *F = (FloatTensor *) m_output[0];
		for(int i=0; i < N; i++)
		{
			(*F)(i,0) = a[2*i];
			(*F)(i,1) = a[2*i+1];
		}

		// Free workspace
		free_1d_double(a);
		free_1d_int(ip);
		free_1d_double(w);
#endif

		delete RI;
	}
	else if (input.nDimension() == 2)
	{
		if(inverse)
		{
			R->select(t_input, 1, 0);
			I->select(t_input, 1, 1);

#ifdef HAVE_OOURAFFT
			// Workspace for Ooura FFT
			int *ip;
			double *w;
			double *a;
			
			// Alloc workspace
			a = alloc_1d_double(2*N);
			ip = alloc_1d_int(2 + (int) sqrt(N + 0.5));
			w = alloc_1d_double(N * 5 / 4);
  
			// Init workspace
			ip[0] = 0;
			for(int i=0; i < N; i++)
			{
				a[2*i] = (*R)(i);
				a[2*i+1] = (*I)(i);
			}

			// Complex Discrete Fourier Transform routine (in inverse mode)
			cdft(N, 1, a, ip, w);

			FloatTensor *F = (FloatTensor *) m_output[0];
			for(int i=0; i < N; i++)
				(*F)(i) = 2.0 * a[i] / N;

			// Free workspace
			free_1d_int(ip);
			free_1d_double(w);
			free_1d_double(a);
#endif
		}
		else
		{
			FloatTensor *RI = new FloatTensor(H, W);
			
			RI->copy(t_input); 

#ifdef HAVE_OOURAFFT
			// Workspace for Ooura FFT
			int *ip;
			double *w;
			double **a;
			int n;

			// Alloc workspace
			a = alloc_2d_double(H, W*2);
			n = MAX(H, W / 2);
			ip = alloc_1d_int(2 + (int) sqrt(n + 0.5));
			n = MAX(H, W) * 3 / 2;
			w = alloc_1d_double(n);

			// Init workspace
			ip[0] = 0;

			for(int i = 0 ; i < H ; i++)
			{
				for(int j = 0 ; j < W ; j++) a[i][j] = (*RI)(i,j);
				for(int j = W ; j < 2*W ; j++) a[i][j] = 0;
			}

			// Complex Discrete Fourier Transform 2D
			cdft2d(H, W, -1, a, NULL, ip, w);

			FloatTensor *F = (FloatTensor *) m_output[0];
			for(int i = 0 ; i < H ; i++)
				for(int j = 0 ; j < W ; j++)
				{
					(*F)(i,j,0) = a[i][2*j];
					(*F)(i,j,1) = a[i][2*j+1];
				}


			// Free workspace
			free_1d_int(ip);
			free_1d_double(w);
			free_2d_double(a);
#endif
			delete RI;
		}
	}
	else if (input.nDimension() == 3)
	{
		if(inverse)
		{
			R->select(t_input, 2, 0);
			I->select(t_input, 2, 1);

#ifdef HAVE_OOURAFFT
			// Workspace for Ooura FFT
			int *ip;
			double *w;
			double **a;
			int n;

			// Alloc workspace
			a = alloc_2d_double(H, W*2);
			n = MAX(H, W / 2);
			ip = alloc_1d_int(2 + (int) sqrt(n + 0.5));
			n = MAX(H, W) * 3 / 2;
			w = alloc_1d_double(n);

			// Init workspace
			ip[0] = 0;

			for(int i = 0 ; i < H ; i++)
				for(int j = 0 ; j < W ; j++)
				{
					a[i][2*j] = (*R)(i,j);
					a[i][2*j+1] = (*I)(i,j);
				}

			// Complex Discrete Fourier Transform 2D (in inverse mode)
			cdft2d(H, W, 1, a, NULL, ip, w);

			//
			FloatTensor *iF = (FloatTensor *) m_output[0];
			double scale = 2.0 / ((double) H*W);
			for(int i = 0 ; i < H ; i++)
				for(int j = 0 ; j < W ; j++)
					(*iF)(i,j) = scale * a[i][j];

			// Free workspace
			free_1d_int(ip);
			free_1d_double(w);
			free_2d_double(a);
#endif
		}
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////

}

