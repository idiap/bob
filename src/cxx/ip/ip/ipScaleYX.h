/**
 * @file cxx/ip/ip/ipScaleYX.h
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef _TORCHVISION_IP_SCALE_YX_H_
#define _TORCHVISION_IP_SCALE_YX_H_

#include "ip/ipCore.h"		// <ipScaleYX> is a <Torch::ipCore>
#include "ip/vision.h"

namespace Torch
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::ipScaleYX
	//	This class is designed to scale an image.
	//	The result is a tensor of the same storage type.
	//
	//	\verbatim
	//
	//	+----+          +--------------+
	//	|XXXX|          |              |         +--+
        //	|XXXX|   ---->  |  ipScale_yx  | ---->   |XX|
	//	|XXXX|          |              |         |XX|
	//	+----+          +--------------+         +--+
	//
        //	image                                 scaled image
	//
	//	\endverbatim
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
