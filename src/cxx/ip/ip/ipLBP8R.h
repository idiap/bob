/**
 * @file cxx/ip/ip/ipLBP8R.h
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
#ifndef _BOBVISION_IP_LBP_8R_H_
#define _BOBVISION_IP_LBP_8R_H_

#include "ip/ipLBP.h"		// <ipLBP8R> is an <ipLBP>
#include "sp/spCores.h"

namespace bob
{
	/////////////////////////////////////////////////////////////////////////
	// bob::ipLBP8R
	//	This class implements LBP8R operators, where R is the radius.
	//	Uses the "Uniform" and "RotInvariant" boolean options.
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class ipLBP8R : public ipLBP
	{
	public:

		// Constructor
		ipLBP8R(int R = 1);

		// Destructor
		virtual ~ipLBP8R();

		// Set the radius value of the LBP operator
		virtual bool		setR(int R);

		// Get the maximum possible label
		virtual int		getMaxLabel();

		/// Loading/Saving the content from files (<em>not the options</em>)
		virtual bool		loadFile(File& file);
		virtual bool		saveFile(File& file) const;

		// Get the ID specific to each spCore - overriden
		virtual int		getID() const { return IP_LBP_8R_ID; }

		/// Constructs an empty spCore of this kind - overriden
		/// (used by \c spCoreManager, this object is automatically deallocated)
		virtual spCore*		getAnInstance() const { return manage(new ipLBP8R()); }

		/////////////////////////////////////////////////////////////////

	protected:

		/////////////////////////////////////////////////////////////////

		/// Process some input tensor (the input is checked, the outputs are allocated) - overriden
		virtual bool		processInput(const Tensor& input);

		/////////////////////////////////////////////////////////////////

	private:

		/////////////////////////////////////////////////////////////////

		// Initialize the conversion table for rotation invariant and uniform LBP patterns
		void			init_lut_RI();
		void			init_lut_U2();
		void			init_lut_U2RI();

		/////////////////////////////////////////////////////////////////
		// Attributes

		float 			m_r;
	};

	// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // REGISTER this spCore to the \c spCoreManager
        const bool ip_lbp_8r_registered = spCoreManager::getInstance().add(
                manage(new ipLBP8R()), "LBP 8R");
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
}

#endif
