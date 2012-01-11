/**
 * @file cxx/ip/ip/ipRotate.h
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
#ifndef _BOBVISION_IP_ROTATE_H_
#define _BOBVISION_IP_ROTATE_H_

#include "ip/ipCore.h"		// <ipRotate> is a <bob::ipCore>
#include "ip/RotationMatrix2D.h"
#include "ip/ipShift.h"
#include "core/Tensor.h"

namespace bob
{
	/////////////////////////////////////////////////////////////////////////
	// bob::ipRotate
	//	This class is designed to rotate an image.
	//	The result is a tensor of the same storage type, but may have a different size.
	//
	//	It implements the rotation by shear also called Paeth rotation.
	//
	//	\verbatim
	//		@inproceedings{paeth:86,
	//		author = {A W Paeth},
	//		title = {A fast algorithm for general raster rotation},
	//		booktitle = {Proceedings on Graphics Interface '86/Vision Interface '86},
	//		year = {1986},
	//		pages = {77--81},
	//		location = {Vancouver, British Columbia, Canada},
	//		publisher = {Canadian Information Processing Society},
	//		}
	//	\endverbatim
	//
	//	The original source code can be found \URL[here]{http://www.codeproject.com/bitmap/rotatebyshear.asp}
	//
	//	- PARAMETERS (name, type, default value, description):
	//		"centerx"	int	0	"Ox coordinate of the rotation center"
	//		"centery"	int	0	"Oy coordinate of the rotation center"
	//		"angle"		double	0.0	"angle in degrees of the rotation"
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class ipRotate : public ipCore
	{
	public:

		// Constructor
		ipRotate();

		// Destructor
		virtual ~ipRotate();

	protected:

		//////////////////////////////////////////////////////////

		/// Check if the input tensor has the right dimensions and type - overriden
		virtual bool		checkInput(const Tensor& input) const;

		/// Allocate (if needed) the output tensors given the input tensor dimensions - overriden
		virtual bool		allocateOutput(const Tensor& input);

		/// Process some input tensor (the input is checked, the outputs are allocated) - overriden
		virtual bool		processInput(const Tensor& input);

		/////////////////////////////////////////////////////////////////

	private:

		/////////////////////////////////////////////////////////////////
		// Auxiliary functions (interpolation and special case rotation)

		void 			HorizSkew(	const ShortTensor& src, ShortTensor& dst,
							int uRow, int iOffset, double dWeight);
		void 			VertSkew(	const ShortTensor& src, ShortTensor& dst,
							int uCol, int iOffset, double dWeight);

		bool			rotateimage0(const ShortTensor& src, ShortTensor& dst);
		bool			rotateimage90(const ShortTensor& src, ShortTensor& dst);
		bool			rotateimage180(const ShortTensor& src, ShortTensor& dst);
		bool			rotateimage270(const ShortTensor& src, ShortTensor& dst);
		bool			rotateimage45(const ShortTensor& src, ShortTensor& dst, double angle);

		/////////////////////////////////////////////////////////////////
		// Attributes

		// Rotation matrix
		RotationMatrix2D	m_rotMatrix;

		// Object to shift the image from the center of rotation
		ipShift			m_shifter;

		// Buffers to store temporary results
		ShortTensor		m_pixmap1;
		ShortTensor		m_pixmap2;
		ShortTensor		m_pixmap3;
	};
}

#endif
