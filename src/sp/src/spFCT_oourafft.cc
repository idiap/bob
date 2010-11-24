#include "sp/spFCT_oourafft.h"
#include "oourafft/ooura.h"

#define MAX(x,y) ((x) > (y) ? (x) : (y))

namespace Torch {

/////////////////////////////////////////////////////////////////////////
// Constructor
spFCT_oourafft::spFCT_oourafft(bool inverse_)
	:	spCore()
{
	inverse = inverse_;
	R = NULL;

	addBOption("verbose", false, "verbose");
}


/////////////////////////////////////////////////////////////////////////
// Destructor
spFCT_oourafft::~spFCT_oourafft()
{
	if(R != NULL) delete R;
}


//////////////////////////////////////////////////////////////////////////
// Check if the input tensor has the right dimensions and type
bool spFCT_oourafft::checkInput(const Tensor& input) const
{
	if (input.nDimension() == 1)
	{
	 	//if(inverse) print("spFCT_oourafft::checkInput() inverse FCT_oourafft 
	 	//  1D ...\n");
		//else print("spFCT_oourafft::checkInput() FCT_oourafft 1D ...\n");

		int N_ = input.size(0);
		unsigned int nn = nexthigher(N_);
		if(N_ != (int) nn) {
			warning("spFCT_oourafft(): size(0) is not a power of 2.");
			return false;
		}
	}
	else if (input.nDimension() == 2)
	{
	 	//if(inverse) print("spFCT_oourafft::checkInput() inverse FCT_oourafft
	 	//  2D ...\n");
		//else print("spFCT_oourafft::checkInput() FCT_oourafft 2D ...\n");

		int N_ = input.size(0);
		unsigned int nn = nexthigher(N_);
		if(N_ != (int) nn) {
			warning("spFCT_oourafft(): size(0) is not a power of 2.");
			return false;
		}

		N_ = input.size(1);
		nn = nexthigher(N_);
		if(N_ != (int) nn) {
			warning("spFCT_oourafft(): size(1) is not a power of 2.");
			return false;
		}
	}
	else
	{
		warning("spFCT_oourafft(): incorrect number of dimensions for the input \
      tensor.");
		return false;
	}

	// OK
	return true;
}


/////////////////////////////////////////////////////////////////////////
// Allocate (if needed) the output tensors given the input tensor dimensions
bool spFCT_oourafft::allocateOutput(const Tensor& input)
{
	bool verbose = getBOption("verbose");

	if (	m_output == 0 )
	{
		cleanup();

		if(R != NULL) delete R;

		if (input.nDimension() == 1)
		{
			if(verbose) print("spFCT_oourafft::allocateOutput() FCT_oourafft \
        1D ...\n");

			N = input.size(0);

			m_n_outputs = 1;
			m_output = new Tensor*[m_n_outputs];
			m_output[0] = new FloatTensor(N);

			R = new FloatTensor(N);
		}
		else if (input.nDimension() == 2)
		{
			if(verbose) print("spFCT_oourafft::allocateOutput() FCT_oourafft \
        2D ...\n");

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
bool spFCT_oourafft::processInput(const Tensor& input)
{
	if (input.nDimension() == 1)
	{
		R->copy(&input);

		if(inverse)
		{
	   	// Declare variables
			int *ip;
			double *w;
			double *a;
      double a0;

			// Allocate C arrays according to the oourafft requirements 
			//    (slightly larger)
			a = alloc_1d_double(N);
			ip = alloc_1d_int(2 + (int) sqrt(N + 0.5));
			w = alloc_1d_double(N * 3 / 2);

			// Set ip[0] such that the cos/sine coefficients are computed and
			// stored in w correctly.
			ip[0] = 0;
			// Copy the float tensor in the C array and save first value
			for(int i=0; i < N; i++) a[i] = (*R)(i);
			a0 = a[0];

      // Compute the inverse FCT_oourafft (second argument set to 1)
			ddct(N, 1, a, ip, w);

			// The output of the oourafft implementation does not use the 
			// scale1/scale2 coefficients. This has to be done manually.
			// This is a bit tricky for the inverse, as the sum needs to be 
			// processed manually by removing the first initial value a0.
			double scale1 = sqrt(1.0 / N);
			double scale2 = sqrt(2.0 / N);
			FloatTensor *F = (FloatTensor *) m_output[0];
			for(int i=0; i < N; i++) (*F)(i) = scale1 * a0 + scale2 * (a[i]-a0);

			// Deallocate memory
			free_1d_int(ip);
			free_1d_double(w);
			free_1d_double(a);
		}
		else
		{
	   	// Declare variables
			int *ip;
			double *w;
			double *a;

			// Allocate C arrays according to the oourafft requirements 
			//    (slightly larger)
			a = alloc_1d_double(N);
			ip = alloc_1d_int(2 + (int) sqrt(N + 0.5));
			w = alloc_1d_double(N * 3 / 2);

			// Set ip[0] such that the cos/sine coefficients are computed and
			// stored in w correctly.
			ip[0] = 0;
			// Copy the float tensor in the C array
			for(int i=0; i < N; i++) a[i] = (*R)(i);

      // Compute the FCT_oourafft (second argument set to -1)
			ddct(N, -1, a, ip, w);

			// The output of the oourafft implementation does not use the 
			// scale1/scale2 coefficients. This has to be done manually.
			// This is done separately for the first coefficients
			FloatTensor *F = (FloatTensor *) m_output[0];
      (*F)(0) = sqrt(1./N) * a[0];
      double scale = sqrt(2./N);
			for(int i=1; i < N; i++) (*F)(i) = scale * a[i];

			// Deallocate memory
			free_1d_double(a);
			free_1d_int(ip);
			free_1d_double(w);
		}
	}
	else if (input.nDimension() == 2)
	{
		R->copy(&input);

		if(inverse)
		{
		  // Allocate C array
			double **a = alloc_2d_double(H, W);

			// Copy the FloatTensor in the C array
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
		   	// Declare variables 
				int *ip, n;
				double *w;

				// Allocate arrays
				n = MAX(H, W / 2);
				ip = alloc_1d_int(2 + (int) sqrt(n + 0.5));
				n = MAX(H, W) * 3 / 2;
				w = alloc_1d_double(n);

				// Initialize first value of the working array to zero. Thus,
				// exponential coefficients will be initialized correctly by oourafft
				ip[0] = 0;

				// Rescale C array
        double sqrt1H = sqrt(1./H); 
        double sqrt2H = sqrt(2./H); 
        double sqrt1W = sqrt(1./W);
        double sqrt2W = sqrt(2./W); 
				for(int i=0; i < H; ++i)
					for(int j=0; j < W; ++j) 
            a[i][j] = a[i][j]*(i==0?sqrt1H:sqrt2H)*(j==0?sqrt1W:sqrt2W);

        // Compute the 2D inverse DCT
				ddct2d(H, W, 1, a, NULL, ip, w);

				// Update the output tensor
				FloatTensor *F = (FloatTensor *) m_output[0];
				for(int i=0; i < H; i++)
					for(int j=0; j < W; j++) (*F)(i,j) = a[i][j];

				// Free dynamically allocated memory
				free_1d_int(ip);
				free_1d_double(w);

			}

			// Free dynamically allocated memory
			free_2d_double(a);
		}
		else
		{
	   	// Allocate C array
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
		   	// Declare variables
				int *ip, n;
				double *w;

				// Allocate arrays
				n = MAX(H, W / 2);
				ip = alloc_1d_int(2 + (int) sqrt(n + 0.5));
				n = MAX(H, W) * 3 / 2;
				w = alloc_1d_double(n);

				// Initialize first value of the working array to zero. Thus,
				// exponential coefficients will be initialized correctly by oourafft
				ip[0] = 0;

				// Compute the 2D DCT
				ddct2d(H, W, -1, a, NULL, ip, w);

				// Update the output tensor
				FloatTensor *F = (FloatTensor *) m_output[0];
        double sqrt1H = sqrt(1./H); 
        double sqrt2H = sqrt(2./H); 
        double sqrt1W = sqrt(1./W);
        double sqrt2W = sqrt(2./W); 
				for(int i=0; i < H; ++i)
					for(int j=0; j < W; ++j) 
            (*F)(i,j) = a[i][j]*(i==0?sqrt1H:sqrt2H)*(j==0?sqrt1W:sqrt2W);

				// Free dynamically allocated memory
				free_1d_int(ip);
				free_1d_double(w);
			}

			// Free dynamically allocated memory
			free_2d_double(a);
		}
	}

	// OK
	return true;
}


}

