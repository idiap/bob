#include "ip/ipSmoothGaussian.h"
#include "core/Tensor.h"

namespace Torch {

/////////////////////////////////////////////////////////////////////////
// Constructor

ipSmoothGaussian::ipSmoothGaussian()
	:	ipCore(),
		m_kernel(0)
{
	addIOption("RadiusX", 1, "Kernel radius on Ox");
	addIOption("RadiusY", 1, "Kernel radius on Oy");
	addDOption("Sigma", 5.0, "Variance of the kernel");
}

/////////////////////////////////////////////////////////////////////////
// Destructor

ipSmoothGaussian::~ipSmoothGaussian()
{
	delete m_kernel;
}

//////////////////////////////////////////////////////////////////////////
// Check if the input tensor has the right dimensions and type

bool ipSmoothGaussian::checkInput(const Tensor& input) const
{
	// Accept only 3D tensors of Torch::Image type
	if (	input.nDimension() != 3 ||
		input.getDatatype() != Tensor::Short)
	{
		return false;
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Allocate (if needed) the output tensors given the input tensor dimensions

bool ipSmoothGaussian::allocateOutput(const Tensor& input)
{
	if (	m_output == 0 ||
		m_output[0]->nDimension() != 3 ||
		m_output[0]->size(0) != input.size(0) ||
		m_output[0]->size(1) != input.size(1) ||
		m_output[0]->size(2) != input.size(2))
	{
		cleanup();

		// Need allocation
		m_n_outputs = 1;
		m_output = new Tensor*[m_n_outputs];
		m_output[0] = new ShortTensor(input.size(0), input.size(1), input.size(2));
		return true;
	}

	return true;
}

/////////////////////////////////////////////////////////////////////////
// Allocate and compute the Gaussian kernel

void ipSmoothGaussian::prepareKernel(int radius_x, int radius_y, double sigma)
{
	// Check if the kernel needs allocation
	if (	m_kernel == 0 ||
		m_kernel->size(0) != (2 * radius_y + 1) ||
		m_kernel->size(1) != (2 * radius_x + 1))
	{
		delete m_kernel;
		m_kernel = new DoubleTensor(2 * radius_y + 1, 2 * radius_x + 1);
	}

	// Compute the kernel
	const double inv_sigma = 1.0  / sigma;
	double sum = 0.0;
 	for (int i = -radius_x; i <= radius_x; i ++)
 		for (int j = -radius_y; j <= radius_y; j ++)
 		{
 			const double weight = exp(- inv_sigma * (i * i + j * j));
 			(*m_kernel)(j + radius_y, i + radius_x) = weight;
    	sum += weight;
 		}

	// Normalize the kernel
 	const double inv_sum = 1.0 / sum;
 	for (int i = -radius_x; i <= radius_x; i ++)
 		for (int j = -radius_y; j <= radius_y; j ++)
 		{
			(*m_kernel)(j + radius_y,	i + radius_x) *= inv_sum;
		}
}

/////////////////////////////////////////////////////////////////////////
// Process some input tensor (the input is checked, the outputs are allocated)

bool ipSmoothGaussian::processInput(const Tensor& input)
{
	// Get the parameters
	const int radius_x = getIOption("RadiusX");
	const int radius_y = getIOption("RadiusY");
	const double sigma = getDOption("Sigma");

	// Allocate and compute the kernel
	prepareKernel(radius_x, radius_y, sigma);

	// Prepare the input and output 3D image tensors
	const ShortTensor* t_input = (ShortTensor*)&input;
	ShortTensor* t_output = (ShortTensor*)m_output[0];

	const int height = input.size(0);
	const int width = input.size(1);
	const int n_planes = input.size(2);

	const int start_x = radius_x;
	const int start_y = radius_y;
	const int stop_x = width - radius_x;
	const int stop_y = height - radius_y;

	// Fill with 0 the output image (to clear boundaries)
	t_output->fill(0);

	// Apply the kernel to the image for each color plane
	for (int p = 0; p < n_planes; p ++)
		for (int y = start_y; y < stop_y; y ++)
			for (int x = start_x; x < stop_x; x ++)
			{
				// Apply the kernel for the <y, x> pixel
				double sum = 0.0;
				for (int yy = -radius_y; yy <= radius_y; yy ++)
					for (int xx = -radius_x; xx <= radius_x; xx ++)
					{
						sum += 	(*m_kernel)(yy + radius_y, xx + radius_x) *	(*t_input)(y + yy, x + xx, p);
					}

				(*t_output)(y,x,p) = FixI(sum);
			}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////

}


