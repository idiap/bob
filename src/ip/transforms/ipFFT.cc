#include "ipFFT.h"

namespace Torch {

/////////////////////////////////////////////////////////////////////////
// Constructor

ipFFT::ipFFT(bool inverse_)
	:	ipCore()
{
	inverse = inverse_;
}

/////////////////////////////////////////////////////////////////////////
// Destructor

ipFFT::~ipFFT()
{
	delete tmp1;
	delete I;
	delete R;
}

unsigned int nexthigher(unsigned int k) 
{
	if (k == 0) return 1;
	k--;	           
	for (int i=1; i<sizeof(unsigned int)*8; i<<=1)
		k = k | k >> i;
	return k+1;
}

//////////////////////////////////////////////////////////////////////////
// Check if the input tensor has the right dimensions and type

bool ipFFT::checkInput(const Tensor& input) const
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
		print("ipFFT::checkInput() assuming FFT 1D ...\n");

	   	if(inverse)
		{
			warning("ipFFT(): impossible to handle inverse mode with 1D input tensor.");
			return false;
		}

		int N_ = input.size(0);

		unsigned int nn = nexthigher(N_); 
		
		if(N_ != (int) nn)
		{
			warning("ipFFT(): size(0) is not a power of 2.");
			return false;
		}
	}
	
	if (input.nDimension() == 2)
	{
	   	if(inverse)
		{
			print("ipFFT::checkInput() assuming inverse FFT 1D ...\n");

			int N_ = input.size(0);
			unsigned int nn = nexthigher(N_); 
			if(N_ != (int) nn)
			{
				warning("ipFFT(): size(0) is not a power of 2.");
				return false;
			}
		}
		else
		{
			print("ipFFT::checkInput() assuming FFT 2D ...\n");

			int N_ = input.size(0);
			unsigned int nn = nexthigher(N_); 
			if(N_ != (int) nn)
			{
				warning("ipFFT(): size(0) is not a power of 2.");
				return false;
			}
			N_ = input.size(1);
			nn = nexthigher(N_); 
			if(N_ != (int) nn)
			{
				warning("ipFFT(): size(1) is not a power of 2.");
				return false;
			}
		}
	}
	
	if (input.nDimension() == 3)
	{
		print("ipFFT::checkInput() assuming inverse FFT 2D ...\n");

	   	if(inverse == false)
		{
			warning("ipFFT(): impossible to handle forward mode with 3D input tensor.");
			return false;
		}

		if(input.size(2) != 2)
		{
			warning("ipFFT(): size(2) is not 2 (necessary to handle real and imag parts).");
			return false;
		}

		int N_ = input.size(0);
		unsigned int nn = nexthigher(N_); 
		if(N_ != (int) nn)
		{
			warning("ipFFT(): size(0) is not a power of 2.");
			return false;
		}
		N_ = input.size(1);
		nn = nexthigher(N_); 
		if(N_ != (int) nn)
		{
			warning("ipFFT(): size(1) is not a power of 2.");
			return false;
		}
	}
	
	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Allocate (if needed) the output tensors given the input tensor dimensions

bool ipFFT::allocateOutput(const Tensor& input)
{
	if (	m_output == 0 )
	{
		cleanup();

		if (input.nDimension() == 1)
		{
			print("ipFFT::allocateOutput() assuming FFT 1D ...\n");

			N = input.size(0);

			m_n_outputs = 1;
			m_output = new Tensor*[m_n_outputs];
			m_output[0] = new FloatTensor(N, 2);
		
			R = new FloatTensor;
			I = new FloatTensor;
			tmp1 = new DoubleTensor(2*N+1);
		}
		else if (input.nDimension() == 2)
		{
		   	if(inverse)
			{
				print("ipFFT::allocateOutput() assuming inverse FFT 1D ...\n");

				N = input.size(0);

				m_n_outputs = 1;
				m_output = new Tensor*[m_n_outputs];
				m_output[0] = new FloatTensor(N);
		
				R = new FloatTensor;
				I = new FloatTensor;
				tmp1 = new DoubleTensor(2*N+1);
			}
			else
			{
				print("ipFFT::allocateOutput() assuming FFT 2D ...\n");

				H = input.size(0);
				W = input.size(0);

				m_n_outputs = 1;
				m_output = new Tensor*[m_n_outputs];
				m_output[0] = new FloatTensor(H,W,2);
		
				R = new FloatTensor;
				I = new FloatTensor;
				tmp1 = new DoubleTensor(2*W+1);
				tmp2 = new DoubleTensor(2*H+1);
				T = new FloatTensor(2*H+1,W);
			}
		}
		else if (input.nDimension() == 3)
		{
			print("ipFFT::allocateOutput() assuming inverse FFT 2D ...\n");

			H = input.size(0);
			W = input.size(0);

			m_n_outputs = 1;
			m_output = new Tensor*[m_n_outputs];
			m_output[0] = new FloatTensor(H,W);
		
			R = new FloatTensor;
			I = new FloatTensor;
			tmp1 = new DoubleTensor(2*W+1);
			tmp2 = new DoubleTensor(2*H+1);
			T = new FloatTensor(2*H+1,W);
		}
	}

	return true;
}

/////////////////////////////////////////////////////////////////////////
// Process some input tensor (the input is checked, the outputs are allocated)

#define SWAP(a,b) tempr=(a);(a)=(b);(b)=tempr

void four1(double *data, int nn, int isign)
{
	int n, mmax, m, j, istep, i;
	double wtemp, wr, wpr, wpi, wi, theta;
	double tempr, tempi;

	n = nn << 1;
	j = 1;
	for (i=1;i<n;i+=2) {
		if (j > i) {
			SWAP(data[j],data[i]);
			SWAP(data[j+1],data[i+1]);
		}
		m = n >> 1;
		while (m >= 2 && j > m) {
			j -= m;
			m >>= 1;
		}
		j += m;
	}
	mmax = 2;
	while (n > mmax) {
		istep = 2*mmax;
		theta = 6.28318530717959/(isign*mmax);
		wtemp = sin(0.5*theta);
		wpr = -2.0*wtemp*wtemp;
		wpi = sin(theta);
		wr = 1.0;
		wi = 0.0;
		for (m=1;m<mmax;m+=2) {
			for (i=m;i<=n;i+=istep) {
				j = i+mmax;
				tempr = wr*data[j]-wi*data[j+1];
				tempi = wr*data[j+1]+wi*data[j];
				data[j] = data[i]-tempr;
				data[j+1] = data[i+1]-tempi;
				data[i] += tempr;
				data[i+1] += tempi;
			}
			wr = (wtemp=wr)*wpr-wi*wpi+wr;
			wi = wi*wpr+wtemp*wpi+wi;
		}
		mmax = istep;
	}
}

/* interlace function

   copy the real and imag part from 2 1D tensors (R and I) with running index [0 .. N-1]
   to a single double-precision data vector with a running index [1 .. N]

	data : [0 R I R I ... R I]

*/
void interlace(int N, double *data_, FloatTensor *R, FloatTensor *I)
{
	double *r_ = &data_[1];
	double *i_ = &data_[2];

	for(int i = 0 ; i < N ; i++)
	{
		*r_ = (*R)(i); 
	  	*i_ = (*I)(i); 

		r_ += 2;
		i_ += 2;
	}
}

/* desinterlace function

   copy the real and imag part from a single double-precision data vector with a running index [1 .. N]
   to a single 2D tensors (F containing the real and imag part as the second dimension) with running index [0 .. N-1] on the first dimension

	data : [0 R I R I ... R I]

	F : [ R I
	      R I
	      ... 
	      R I ]

*/
void desinterlace(int N, double *data_, FloatTensor *F)
{
	float *data_r_ = (F->t->storage->data+F->t->storageOffset);
	float *data_i_ = &((F->t->storage->data+F->t->storageOffset)[F->t->stride[1]]);

	double *r_ = &data_[1];
	double *i_ = &data_[2];

	for(int i = 0 ; i < N ; i++)
	{
		data_r_[i] = *r_;
		data_i_[i] = *i_;

		r_ += 2;
		i_ += 2;
	}
}

/* desinterlace function

   copy the real part ONLY from a single double-precision data vector with a running index [1 .. N]
   to a single 1D tensor (F containing the real part) with running index [0 .. N-1]

   a normalisation factor is given as this function is mainly used for inverse purposes

	data : [0 R I R I ... R I]

	F : [ R
	      R
	      ... 
	      R ]

*/
void desinterlace_inverse(int N, double *data_, FloatTensor *F, double norm)
{
	float *data_r_ = (F->t->storage->data+F->t->storageOffset);

	double *r_ = &data_[1];

	for(int i = 0 ; i < N ; i++)
	{
		data_r_[i] = *r_ / norm;

		r_ += 2;
	}
}

bool ipFFT::processInput(const Tensor& input)
{
	const FloatTensor* t_input = (FloatTensor*)&input;

	if (input.nDimension() == 1)
	{
		FloatTensor *RI = new FloatTensor(N, 2);

		R->select(RI, 1, 0);
		R->copy(t_input);
		I->select(RI, 1, 1);
		I->fill(0.0);

		double *data_ = (double *)(tmp1->t->storage->data+tmp1->t->storageOffset);
		interlace(N, data_, R, I);
		
		four1(data_, N, 1);

		FloatTensor *F = (FloatTensor *) m_output[0];
		desinterlace(N, data_, F);

		delete RI;
	}
	else if (input.nDimension() == 2)
	{
		if(inverse)
		{
			R->select(t_input, 1, 0);
			I->select(t_input, 1, 1);

			double *data_ = (double *)(tmp1->t->storage->data+tmp1->t->storageOffset);
			interlace(N, data_, R, I);
			
			four1(data_, N, -1);

			FloatTensor *iF = (FloatTensor *) m_output[0];
			desinterlace_inverse(N, data_, iF, (double) N);
		}
		else
		{
			FloatTensor *RI = new FloatTensor(H, W, 2);
			R->select(RI, 2, 0);
			R->copy(t_input); 
			I->select(RI, 2, 1);
			I->fill(0.0);

			FloatTensor *subR = new FloatTensor();
			FloatTensor *subI = new FloatTensor();
			FloatTensor *subT = new FloatTensor();
			FloatTensor *subF = new FloatTensor();

			T->fill(0.0);

			double *data_ = (double *)(tmp1->t->storage->data+tmp1->t->storageOffset);

        		for(int i=1 ; i <= H ; i++) 
			{
			   	subR->select(R, 0, i-1);
			   	subI->select(I, 0, i-1);

			   	interlace(W, data_, subR, subI);

        			four1(data_, W, 1);
			    
        		    	for (int j=1 ; j <= W ; j++) 
			    	{
        		        	(*T)(i*2-1,j-1) = (*tmp1)(j*2-1);
        		        	(*T)(i*2,j-1) = (*tmp1)(j*2);
        		    	}
        		}

			data_ = (double *)(tmp2->t->storage->data+tmp2->t->storageOffset);

			FloatTensor *F = (FloatTensor *) m_output[0];

        		for(int i=1 ; i <= W ; i++) 
			{
        		    	for (int j=1 ; j <= H ; j++) 
			    	{
        		        	(*tmp2)(j*2-1) = (*T)(j*2-1,i-1);
        		        	(*tmp2)(j*2) = (*T)(j*2,i-1);
        		    	}

        			four1(data_, H, 1);
			    
			   	subF->select(F, 1, i-1);
				desinterlace(H, data_, subF);
        		}

			delete subF;
			delete subT;
			delete subR;
			delete subI;
			delete RI;
		}
	}
	else if (input.nDimension() == 3)
	{
		if(inverse)
		{
			R->select(t_input, 2, 0);
			I->select(t_input, 2, 1);

			FloatTensor *subR = new FloatTensor();
			FloatTensor *subI = new FloatTensor();
			FloatTensor *subT = new FloatTensor();
			FloatTensor *subF = new FloatTensor();

			T->fill(0.0);

			double *data_ = (double *)(tmp1->t->storage->data+tmp1->t->storageOffset);

        		for(int i=1 ; i <= H ; i++) 
			{
			   	subR->select(R, 0, i-1);
			   	subI->select(I, 0, i-1);

			   	interlace(W, data_, subR, subI);

        			four1(data_, W, -1);
			    
        		    	for (int j=1 ; j <= W ; j++) 
			    	{
        		        	(*T)(i*2-1,j-1) = (*tmp1)(j*2-1);
        		        	(*T)(i*2,j-1) = (*tmp1)(j*2);
        		    	}
        		}

			FloatTensor *iF = (FloatTensor *) m_output[0];

			data_ = (double *)(tmp2->t->storage->data+tmp2->t->storageOffset);

        		for(int i=1 ; i <= W ; i++) 
			{
        		    	for (int j=1 ; j <= H ; j++) 
			    	{
        		        	(*tmp2)(j*2-1) = (*T)(j*2-1,i-1);
        		        	(*tmp2)(j*2) = (*T)(j*2,i-1);
        		    	}

        			four1(data_, H, -1);
			    
			   	subF->select(iF, 1, i-1);
				desinterlace_inverse(H, data_, subF, (double) (H*W));
        		}

			delete subF;
			delete subT;
			delete subR;
			delete subI;
		}
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////

}

