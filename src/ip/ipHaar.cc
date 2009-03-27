#include "ipHaar.h"

namespace Torch {

/////////////////////////////////////////////////////////////////////////
// Constructor

ipHaar::ipHaar(int width_, int height_, int type_, int x_, int y_, int w_, int h_)
	:	ipCore()
{
	width = width_;
	height = height_;
	type = type_;
	x = x_;
	y = y_;
	w = w_;
	h = h_;

  	t_ = new DoubleTensor();
  	t__ = new DoubleTensor();

  	print("ipHaar() Type-%d (%d-%d) [%dx%d]\n", type, x, y, w, h);

	if(x+w >= width) Torch::fatalerror("ipHaar() incorrect w size");
	if(y+h >= height) Torch::fatalerror("ipHaar() incorrect h size");
}

/////////////////////////////////////////////////////////////////////////
// Destructor

ipHaar::~ipHaar()
{
   	delete t_;
	delete t__;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Check if the input tensor has the right dimensions and type - overriden

bool ipHaar::checkInput(const Tensor& input) const
{
	if(input.nDimension() != 2)
	{
	   	Torch::error("ipHaar::checkInput() input should be 2D");
		return false;
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Allocate (if needed) the output tensors given the input tensor dimensions

bool ipHaar::allocateOutput(const Tensor& input)
{
	// Allocate the output if needed
	if (m_output == 0)
	{
	        m_n_outputs = 1;
                m_output = new Tensor*[m_n_outputs];
                m_output[0] = new DoubleTensor(1);
	}
	return true;
}

bool ipHaar::processInput(const Tensor& input)
{
	const DoubleTensor* t_input = (DoubleTensor*)&input;
	DoubleTensor* t_output = (DoubleTensor*)m_output[0];

  	//print("Tensor (n_dim=%d size=[%dx%d]):\n", t_input->nDimension(), t_input->size(0), t_input->size(1));
  	//t_input->print("input");

	double white_sum = 0;
	double black_sum = 0;
	
	if(type == 1)
	{
	   	// Vertical Haar (2 bands)
  		//print("Haar 1 %d-%d %dx%d:\n", x, y, w, h);

  		t_->narrow(t_input, 0, y, h);
  		t__->narrow(t_, 1, x, w/2);
		white_sum = (double) t__->sum();

  		//t_->narrow(t_input, 0, y, h);
  		t__->narrow(t_, 1, x+w/2, w/2);
		black_sum = (double) t__->sum();
	}
	else if(type == 2)
	{
	   	// Horizontal Haar (2 bands)
  		//print("Haar 2 %d-%d %dx%d:\n", x, y, w, h);

  		t_->narrow(t_input, 0, y, h/2);
  		t__->narrow(t_, 1, x, w);
		white_sum = (double) t__->sum();

  		t_->narrow(t_input, 0, y+h/2, h/2);
  		t__->narrow(t_, 1, x, w);
		black_sum = (double) t__->sum();
	}
	else if(type == 3)
	{
	   	// Vertical Haar (3 bands)
  		//print("Haar 3 %d-%d %dx%d:\n", x, y, w, h);

  		t_->narrow(t_input, 0, y, h);
  		t__->narrow(t_, 1, x, w/3);
		white_sum = (double) t__->sum();

  		t__->narrow(t_, 1, x+w/3, w/3);
		black_sum = 2 * (double) t__->sum();

  		t__->narrow(t_, 1, x+2*w/3, w/3);
		white_sum += (double) t__->sum();
	}
	else if(type == 4)
	{
	   	// Horizontal Haar (3 bands)
  		//print("Haar 4 %d-%d %dx%d:\n", x, y, w, h);

  		t_->narrow(t_input, 0, y, h/3);
  		t__->narrow(t_, 1, x, w);
		white_sum = (double) t__->sum();

  		t_->narrow(t_input, 0, y+h/3, h/3);
  		t__->narrow(t_, 1, x, w);
		black_sum = 2 * (double) t__->sum();

  		t_->narrow(t_input, 0, y+2*h/3, h/3);
  		t__->narrow(t_, 1, x, w);
		white_sum += (double) t__->sum();
	}
	else if(type == 5)
	{
	   	// Central Haar
  		//print("Haar 5 %d-%d %dx%d:\n", x, y, w, h);

  		t_->narrow(t_input, 0, y, h);
  		t__->narrow(t_, 1, x, w);
		white_sum = (double) t__->sum();

  		t_->narrow(t_input, 0, y+h/3, h/3);
  		t__->narrow(t_, 1, x+w/3, w/3);

		black_sum = 2 * (double) t__->sum();
	}
	else if(type == 6)
	{
	   	// Diagonal Haar
  		//print("Haar 6 %d-%d %dx%d:\n", x, y, w, h);

  		t_->narrow(t_input, 0, y, h);
  		t__->narrow(t_, 1, x, w);
		white_sum = (double) t__->sum();

  		t_->narrow(t_input, 0, y, h/2);
  		t__->narrow(t_, 1, x+w/2, w/2);
		black_sum = 2 * (double) t__->sum();

  		t_->narrow(t_input, 0, y+h/2, h/2);
  		t__->narrow(t_, 1, x, w/2);
		black_sum += 2 * (double) t__->sum();
	}
	else
	{
		Torch::warning("ipHaar::processInput() not implemented for type %d", type);
	}

	(*t_output)(0) = white_sum - black_sum;

	return true;
}

bool ipHaar::saveFile(File& file) const
{
  	print("ipHaar()::saveFile()\n");
  	print("   Type = %d\n", type);
  	print("   X-Y = (%d-%d)\n", x, y);
  	print("   WxH = [%dx%d]\n", w, h);

	return true;
}
		
/////////////////////////////////////////////////////////////////////////

}
