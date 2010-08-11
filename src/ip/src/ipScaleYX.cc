#include "ip/ipScaleYX.h"
#include "core/Tensor.h"

namespace Torch {

/////////////////////////////////////////////////////////////////////////
// Constructor

ipScaleYX::ipScaleYX()
	:	ipCore(),
		m_buffer(0),
		m_buffer_size(0)
{
	addIOption("width", 0, "width of the scaled tensor");
	addIOption("height", 0,	"height of the scaled tensor");
}

/////////////////////////////////////////////////////////////////////////
// Destructor

ipScaleYX::~ipScaleYX()
{
	delete[] m_buffer;
}

//////////////////////////////////////////////////////////////////////////
// Check if the input tensor has the right dimensions and type

bool ipScaleYX::checkInput(const Tensor& input) const
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

bool ipScaleYX::allocateOutput(const Tensor& input)
{
	const int out_width = getIOption("width");
	const int out_height = getIOption("height");

	if (	m_output == 0 ||
		m_output[0]->nDimension() != 3 ||
		m_output[0]->size(0) != out_height ||
		m_output[0]->size(1) != out_width ||
		m_output[0]->size(2) != input.size(2))
	{
		cleanup();

		// Need allocation
		m_n_outputs = 1;
		m_output = new Tensor*[m_n_outputs];
		m_output[0] = new ShortTensor(out_height, out_width, input.size(2));
		return true;
	}

	return true;
}

/////////////////////////////////////////////////////////////////////////
// Process some input tensor (the input is checked, the outputs are allocated)

bool ipScaleYX::processInput(const Tensor& input)
{
	const int out_width = getIOption("width");
	const int out_height = getIOption("height");

	const ShortTensor* t_input = (ShortTensor*)&input;
	ShortTensor* t_output = (ShortTensor*)m_output[0];

	// Prepare the arrays to work with
	const short* src = t_input->t->storage->data + t_input->t->storageOffset;
	short* dst = t_output->t->storage->data + t_output->t->storageOffset;

	// Prepare the input/output dimensions
	const int in_height = input.size(0);
	const int in_width = input.size(1);
	const int n_planes = input.size(2);

	const int in_stride_h = t_input->t->stride[0];	// height
	const int in_stride_w = t_input->t->stride[1];	// width
	const int in_stride_p = t_input->t->stride[2];	// no planes

	const int out_stride_h = t_output->t->stride[0];	// height
	const int out_stride_w = t_output->t->stride[1];	// width
	const int out_stride_p = t_output->t->stride[2];	// no planes

	// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	// An index for the 3D tensor is: [y * stride_h + x * stride_w + p * stride_p]
	// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	// Allocate buffer (if needed)
	if (m_buffer_size != n_planes * (3 * in_width + max(in_width, out_width)))
	{
		m_buffer_size = n_planes * (3 * in_width + max(in_width, out_width));
		delete[] m_buffer;
		m_buffer = new float[m_buffer_size];
	}

	// Initialize buffer vectors
	for (int i = 0; i < m_buffer_size; i ++)
	{
		m_buffer[i] = 0.0f;
	}

	float* x_vector = &m_buffer[0];
	float* y_vector = &m_buffer[2 * in_width * n_planes];
	float* scale_scanline = &m_buffer[3 * in_width * n_planes];

	float* scanline = 0;
	if (in_height != out_height)
	{
		scanline = &m_buffer[in_width * n_planes];
	}
   	else
   	{
   		scanline = x_vector;
   	}

	// TODO: need coments here (for every step of the algorithm)!
	// What is the name of this algorithm, a website, some pseudo-code and online explanations?!!
	// Scale image, first scale the Oy axis, then Ox one

	const float x_scale_const = (out_width + 0.0f) / (in_width + 0.0f);
	const float y_scale_const = (out_height + 0.0f) / (in_height + 0.0f);
	float y_scale = y_scale_const;

	const float max_pixel = 255.0f;

	int i_row = 0;	// pointing to the current input line to be considered
	int next_row = 1;

	// Find the scaled image for each line
	for (int y = 0; y < out_height; y ++)
	{
		////////////////////////////////////////////////////////////
		// Oy direction
		////////////////////////////////////////////////////////////

		if (out_height == in_height)
      		{
        		// Read a new scanline
			for (int p = 0; p < n_planes; p ++)
        		{
        			copy(	src + (i_row * in_stride_h + p * in_stride_p), in_stride_w, // source
					x_vector + p, n_planes, // destination
					in_width); // size
        		}
        		i_row ++;
      		}
    		else
      		{
      			float y_span = 1.0f;

      			// Scale Y direction
        		while (y_scale < y_span)
        		{
        			if (next_row && (i_row < in_height))
            			{
              				// Read a new scanline
              				for (int p = 0; p < n_planes; p ++)
					{
						copy(	src + (i_row * in_stride_h + p * in_stride_p), in_stride_w, // size
							x_vector + p, n_planes, // destination
							in_width); // size
					}
					i_row ++;
            			}

				// <y_vector> = <y_vector> + y_scale * <x_vector>
				add(	y_vector, x_vector, y_scale, // sources
					y_vector, // destination
					n_planes * in_width); // size

          			y_span -= y_scale;
          			y_scale = y_scale_const;
          			next_row = 1;
        		}

        		if (next_row && (i_row < in_height))
          		{
            			// Read a new scanline
            			for (int p = 0; p < n_planes; p ++)
				{
					copy(	src + (i_row * in_stride_h + p * in_stride_p), in_stride_w, // source
						x_vector + p, n_planes,	// destination
						in_width); // size
				}

            			i_row ++;
				next_row = 0;
          		}

			// <scanline> = <y_vector> + y_span * <x_vector>
			add(	y_vector, x_vector, y_span, // sources
				scanline, // destination
				n_planes * in_width); // size

			// <scanline> = min(<scanline>, max pixel value)
			const int size = n_planes * in_width;
			for (int k = 0; k < size; k ++)
			{
				scanline[k] = min(scanline[k], max_pixel);
			}

			// <y_vector> = 0
			for (int k = 0; k < size; k ++)
			{
				y_vector[k] = 0.0f;
			}

			// ?!
        		y_scale -= y_span;
        		if (y_scale <= 0.0f)
          		{
            			y_scale = y_scale_const;
            			next_row = 1;
          		}
      		}

		////////////////////////////////////////////////////////////
		// Ox direction
		////////////////////////////////////////////////////////////

      		if (out_width == in_width)
      		{
        		// Transfer scanline to scaled image
        		for (int p = 0; p < n_planes; p ++)
        		{
				copy(	scanline + p, n_planes, // source
					dst + (y * out_stride_h + p * out_stride_p), out_stride_w, // destination
					out_width); // size
        		}
      		}
    		else
      		{
        		// Scale X direction.
        		int next_column = 0;
        		float x_span = 1.0f;

        		for (int p = 0; p < n_planes; p ++)
        		{
        			float* s = scanline + p;
        			float* t = scale_scanline + p;
        			float pixel = 0.0f;

        			for (int x = 0; x < in_width; x ++)
				{
					float x_scale = x_scale_const;

					while (x_scale >= x_span)
					{
						if (next_column)
						{
							pixel = 0.0f;
							t += n_planes;
						}

						pixel += x_span * (*s);
						*t = min(pixel, max_pixel);
						x_scale -= x_span;
						x_span = 1.0f;
						next_column = 1;
					}

					if (x_scale > 0.0f)
					{
						if (next_column)
						{
							pixel = 0.0f;
							next_column = 0;
							t += n_planes;
						}

						pixel += x_scale * (*s);
						x_span -= x_scale;
					}
					s += n_planes;
				}

				if (x_span > 0.0f)
				{
					s -= n_planes;
					pixel += x_span * (*s);
				}

				if (!next_column && ((t - scale_scanline) < n_planes * out_width))
				{
					*t = min(pixel, max_pixel);
				}
        		}

			// Transfer scanline to scaled image
			for (int p = 0; p < n_planes; p ++)
        		{
				copy(	scale_scanline + p, n_planes, // source
					dst + (y * out_stride_h + p * out_stride_p), out_stride_w, // destination
					out_width); // size
        		}
    		}
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Copy a vector to another one, both having different increments

void ipScaleYX::copy(	const short* src, int delta_src,
			float* dst, int delta_dst,
			int count)
{
	for (int i = 0; i < count; i ++, src += delta_src, dst += delta_dst)
	{
		*dst = *src;
	}
}

void ipScaleYX::copy(	const float* src, int delta_src,
			short* dst, int delta_dst,
			int count)
{
	for (int i = 0; i < count; i ++, src += delta_src, dst += delta_dst)
	{
		*dst = (short)(*src);
	}
}

/////////////////////////////////////////////////////////////////////////
// Add two vectors and copy result to another one
//	<dst> = <src1> + coef * <src2>

void ipScaleYX::add(	const float* src1, const float* src2, float coef,
			float* dst,
			int count)
{
	for (int i = 0; i < count; i ++)
	{
		dst[i] = src1[i] + coef * src2[i];
	}
}

/////////////////////////////////////////////////////////////////////////

}
