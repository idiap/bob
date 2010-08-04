#include "ipLBPBitmap.h"
#include "ipLBP.h"
#include "core/Tensor.h"

namespace Torch {

/////////////////////////////////////////////////////////////////////////
// Constructor

ipLBPBitmap::ipLBPBitmap(ipLBP* ip_lbp)
	:	ipCore(),
		m_ip_lbp(ip_lbp)
{
}

/////////////////////////////////////
// Destructor

ipLBPBitmap::~ipLBPBitmap()
{
}

//////////////////////////////////////////////////////////////////////////
// Change the ipLBP to use

bool ipLBPBitmap::setIpLBP(ipLBP* ip_lbp)
{
	if (ip_lbp == 0)
	{
		return false;
	}

	m_ip_lbp = ip_lbp;
	return true;
}

//////////////////////////////////////////////////////////////////////////
// Check if the input tensor has the right dimensions and type

bool ipLBPBitmap::checkInput(const Tensor& input) const
{
	if (m_ip_lbp == 0)
	{
		print("ipLBPBitmap::checkInput - invalid ipLBP!\n");
		return false;
	}

	// Should be type independent.
	// Assume it is 2D now, no checking.

	// Accept only 3D tensors of Torch::Image type
	//if (	input.nDimension() != 3 ||
	//		input.getDatatype() != Tensor::Short)
	//{
	//	return false;
	//}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Allocate (if needed) the output tensors given the input tensor dimensions

bool ipLBPBitmap::allocateOutput(const Tensor& input)
{
	const int maxLabel = m_ip_lbp->getMaxLabel();

	if (	m_output == 0 ||
		m_output[0]->nDimension() != 3 ||
		m_output[0]->size(0) != input.size(0) ||
		m_output[0]->size(1) != input.size(1) ||
		m_output[0]->size(2) != maxLabel)
	{
		cleanup();

		// Need allocation
		m_n_outputs = 1;
		m_output = new Tensor*[m_n_outputs];
		m_output[0] = new CharTensor(input.size(0), input.size(1), maxLabel);

		return true;
	}

	return true;
}

bool ipLBPBitmap::processInput(const Tensor& input)
{
	CharTensor* t_output = (CharTensor*)m_output[0];
	t_output->fill(0x00);

	const int R = m_ip_lbp->getR();
	for (int x = R; x < input.size(1) - R; x ++)
		for (int y = R; y < input.size(0) - R; y ++)
		{
			if (m_ip_lbp->setXY(x, y) == false)
				return false;
			if (m_ip_lbp->process(input) == false)
				return false;
			t_output->set(y, x, m_ip_lbp->getLBP(), 0x01);
		}

	return true;
}

}



