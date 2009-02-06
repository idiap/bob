#ifndef _TORCHVISION_IP_SCALE_YX_H_
#define _TORCHVISION_IP_SCALE_YX_H_

#include "ipCore.h"		// <ipScaleYX> is a <Torch::ipCore>

namespace Torch
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::ipScaleYX
	//	This class is designed to scale an image.
	//	The result is a tensor of the same storage type.
	//
	//	\begin{verbatim}
	//
	//	+----+          +--------------+
	//	|XXXX|          |              |         +--+
        //	|XXXX|   ---->  |  ipScale_yx  | ---->   |XX|
	//	|XXXX|          |              |         |XX|
	//	+----+          +--------------+         +--+
	//
        //	image                                 scaled image
	//
	//	\end{verbatim}
	//
	//	- PARAMETERS (name, type, default value, description):
	//		"width"		int	0	"width of the scaled tensor"
	//		"height"	int	0	"height of the scaled tensor"
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class ipScaleYX : public ipCore
	{
	public:

		// Constructor
		ipScaleYX();

		// Destructor
		virtual ~ipScaleYX();

	protected:

		//////////////////////////////////////////////////////////

		/// Check if the input tensor has the right dimensions and type - overriden
		virtual bool		checkInput(const Tensor& input) const;

		/// Allocate (if needed) the output tensors given the input tensor dimensions - overriden
		virtual bool		allocateOutput(const Tensor& input);

		/// Process some input tensor (the input is checked, the outputs are allocated) - overriden
		virtual bool		processInput(const Tensor& input);

		//////////////////////////////////////////////////////////

	private:

		/////////////////////////////////////////////////////////////////

		// Copy a vector to another one, both having different increments
		void			copy(	const short* src, int delta_src,
						float* dst, int delta_dst,
						int count);
		void			copy(	const float* src, int delta_src,
						short* dst, int delta_dst,
						int count);

		// Add two vectors and copy result to another one
		//	<dst> = <src1> + coef * <src2>
		void			add(	const float* src1, const float* src2, float coef,
						float* dst,
						int count);

		/////////////////////////////////////////////////////////////////
		// Attributes

		sSize			m_outputSize;	// Size of the output tensors
		float*			m_buffer;
		int			m_buffer_size;
	};
}

#endif
