/**
 * @file cxx/old/ap/ap/apCore.h
 * @date Wed Apr 6 14:49:40 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
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
#ifndef APCORE_INC
#define APCORE_INC

#include "sp/spCore.h"

namespace Torch
{

/**
 * \defgroup libap_api libAP API
 * @{
 *
 *  The libAP API.
 */

	class apCore : public spCore
	{
	public:
		/// Constructor
		apCore();

		/// Destructor
		virtual ~apCore();

		/// Change the input audio size
		virtual bool		setAudioSize(int new_length);

		/// Retrieve the input audio size
		int			getAudioSize() const;

	protected:

		/////////////////////////////////////////////
		/// Attributes

		int			m_audioSize;	// Will process only inputs of this size!
	};

/**
 * @}
 */

}


/**
@page libAP AP: an Audio Processing module

@section intro Introduction

AP is the Audio Processing module of Torch.

@section api Documentation
- @ref libap_api "libAP API"

*/

#endif
