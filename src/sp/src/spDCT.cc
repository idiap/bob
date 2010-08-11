#include "sp/spDCT.h"
#include "sp/ooura.h"

/**
 * \addtogroup libsp_api libSP API
 * @{
 *
 *  The libSP API.
 */
#define MAX(x,y) ((x) > (y) ? (x) : (y))

namespace Torch {

/////////////////////////////////////////////////////////////////////////
// Constructor

spDCT::spDCT(bool inverse_)
	:	spCore()
{
	inverse = inverse_;
	R = NULL;

	addBOption("verbose", false, "verbose");
}

/////////////////////////////////////////////////////////////////////////
// Destructor

spDCT::~spDCT()
{
	if(R != NULL) delete R;
}

//////////////////////////////////////////////////////////////////////////
// Check if the input tensor has the right dimensions and type

bool spDCT::checkInput(const Tensor& input) const
{
	// Accept only tensors of Torch::Float
	//if (input.getDatatype() != Tensor::Float) return false;

	if (input.nDimension() == 1)
	{
	   	//if(inverse) print("spDCT::checkInput() inverse DCT 1D ...\n");
		//else print("spDCT::checkInput() DCT 1D ...\n");

		int N_ = input.size(0);

		unsigned int nn = nexthigher(N_);

		if(N_ != (int) nn)
		{
			warning("spDCT(): size(0) is not a power of 2.");
			return false;
		}
	}
	else if (input.nDimension() == 2)
	{
	   	//if(inverse) print("spDCT::checkInput() inverse DCT 2D ...\n");
		//else print("spDCT::checkInput() DCT 2D ...\n");

		int N_ = input.size(0);
		unsigned int nn = nexthigher(N_);
		if(N_ != (int) nn)
		{
			warning("spDCT(): size(0) is not a power of 2.");
			return false;
		}
		N_ = input.size(1);
		nn = nexthigher(N_);
		if(N_ != (int) nn)
		{
			warning("spDCT(): size(1) is not a power of 2.");
			return false;
		}
	}
	else
	{
		warning("spDCT(): incorrect number of dimensions for the input tensor.");
		return false;
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Allocate (if needed) the output tensors given the input tensor dimensions

bool spDCT::allocateOutput(const Tensor& input)
{
	bool verbose = getBOption("verbose");

	if (	m_output == 0 )
	{
		cleanup();

		if(R != NULL) delete R;

		if (input.nDimension() == 1)
		{
			if(verbose) print("spDCT::allocateOutput() DCT 1D ...\n");

			N = input.size(0);

			m_n_outputs = 1;
			m_output = new Tensor*[m_n_outputs];
			m_output[0] = new FloatTensor(N);

			R = new FloatTensor(N);
		}
		else if (input.nDimension() == 2)
		{
			if(verbose) print("spDCT::allocateOutput() DCT 2D ...\n");

			H = input.size(0);
			W = input.size(1);

			m_n_outputs = 1;
			m_output = new Tensor*[m_n_outputs];
			m_output[0] = new FloatTensor(H,W);

			R = new FloatTensor(H,W);
		}
	}

	return true;
}

/////////////////////////////////////////////////////////////////////////
// Process some input tensor (the input is checked, the outputs are allocated)

bool spDCT::processInput(const Tensor& input)
{
	//const FloatTensor* t_input = (FloatTensor*)&input;

	if (input.nDimension() == 1)
	{
		//R->copy(t_input);
		R->copy(&input);

#ifdef HAVE_OOURAFFT
		if(inverse)
		{
		   	//
			int *ip;
			double *w;
			double *a;

			//
			a = alloc_1d_double(N);
			ip = alloc_1d_int(2 + (int) sqrt(N + 0.5));
			w = alloc_1d_double(N * 3 / 2);

			//
			ip[0] = 0;

			//
			for(int i=0; i < N; i++) a[i] = (*R)(i);

			a[0] *= 0.5;

			ddct(N, 1, a, ip, w);

			//
			double scale = 2.0 / N;
			FloatTensor *F = (FloatTensor *) m_output[0];
			for(int i=0; i < N; i++) (*F)(i) = scale * a[i];

			//
			free_1d_int(ip);
			free_1d_double(w);
			free_1d_double(a);
		}
		else
		{
		   	//
			int *ip;
			double *w;
			double *a;

			//
			a = alloc_1d_double(N);
			ip = alloc_1d_int(2 + (int) sqrt(N + 0.5));
			w = alloc_1d_double(N * 3 / 2);

			//
			ip[0] = 0;
			for(int i=0; i < N; i++) a[i] = (*R)(i);

			//
			ddct(N, -1, a, ip, w);

			//
			FloatTensor *F = (FloatTensor *) m_output[0];
			for(int i=0; i < N; i++) (*F)(i) = a[i];

			//
			free_1d_double(a);
			free_1d_int(ip);
			free_1d_double(w);
		}
#endif
	}
	else if (input.nDimension() == 2)
	{
		//R->copy(t_input);
		R->copy(&input);

#ifdef HAVE_OOURAFFT
		if(inverse)
		{
		   	//
			double **a = alloc_2d_double(H, W);

			//
			for(int i=0; i < H; i++)
				for(int j=0; j < W; j++) a[i][j] = (*R)(i,j);

		   	if(W == 8 && H == 8)
			{
    				ddct8x8s(1, a);

				FloatTensor *F = (FloatTensor *) m_output[0];
				for(int i=0; i < H; i++)
					for(int j=0; j < W; j++) (*F)(i,j) = a[i][j];
			}
			else if(W == 16 && H == 16)
			{
    				ddct16x16s(1, a);

				FloatTensor *F = (FloatTensor *) m_output[0];
				for(int i=0; i < H; i++)
					for(int j=0; j < W; j++) (*F)(i,j) = a[i][j];
			}
			else
			{
			   	//
				int *ip, n;
				double *w;

				//
				n = MAX(H, W / 2);
				ip = alloc_1d_int(2 + (int) sqrt(n + 0.5));
				n = MAX(H, W) * 3 / 2;
				w = alloc_1d_double(n);

				//
				ip[0] = 0;

				for (int i = 0; i <= H - 1; i++) a[i][0] *= 0.5;
				for (int i = 0; i <= W - 1; i++) a[0][i] *= 0.5;

				//
				ddct2d(H, W, 1, a, NULL, ip, w);

				//
				double scale = 4.0 / (H * W);
				FloatTensor *F = (FloatTensor *) m_output[0];
				for(int i=0; i < H; i++)
					for(int j=0; j < W; j++) (*F)(i,j) = scale * a[i][j];

				//
				free_1d_int(ip);
				free_1d_double(w);

			}

			//
			free_2d_double(a);
		}
		else
		{
		   	//
			double **a = alloc_2d_double(H, W);

			//
			for(int i=0; i < H; i++)
				for(int j=0; j < W; j++) a[i][j] = (*R)(i,j);

		   	if(W == 8 && H == 8)
			{
    				ddct8x8s(-1, a);

				FloatTensor *F = (FloatTensor *) m_output[0];
				for(int i=0; i < H; i++)
					for(int j=0; j < W; j++) (*F)(i,j) = a[i][j];
			}
			else if(W == 16 && H == 16)
			{
    				ddct16x16s(-1, a);

				FloatTensor *F = (FloatTensor *) m_output[0];
				for(int i=0; i < H; i++)
					for(int j=0; j < W; j++) (*F)(i,j) = a[i][j];
			}
			else
			{
			   	//
				int *ip, n;
				double *w;

				//
				n = MAX(H, W / 2);
				ip = alloc_1d_int(2 + (int) sqrt(n + 0.5));
				n = MAX(H, W) * 3 / 2;
				w = alloc_1d_double(n);

				//
				ip[0] = 0;

				//
				ddct2d(H, W, -1, a, NULL, ip, w);

				//
				FloatTensor *F = (FloatTensor *) m_output[0];
				for(int i=0; i < H; i++)
					for(int j=0; j < W; j++) (*F)(i,j) = a[i][j];

				//
				free_1d_int(ip);
				free_1d_double(w);

			}

			//
			free_2d_double(a);
		}
#endif
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////

}

/**
 * @}
 */

