/**
 * @file cxx/sp/sp/spCoreChain.h
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
#ifndef SPCORE_CHAIN_INC
#define SPCORE_CHAIN_INC

#include "sp/spCore.h"

namespace bob
{
/**
 * \ingroup libsp_api
 * @{
 *
 */

	//////////////////////////////////////////////////////////////////////////////////////
	// bob::spCoreChain:
	//      - process some tensor given a list of spCores
	//
	// TODO: doxygen header!
	//////////////////////////////////////////////////////////////////////////////////////

	class spCoreChain : public spCore
	{
	public:
		/// Constructor
		spCoreChain();

		/// Destructor
		virtual ~spCoreChain();

		/// Change the region of the input tensor to process
		virtual void		setRegion(const TensorRegion& region);

		/// Change the model size (if used with some machine)
		virtual void		setModelSize(const TensorSize& modelSize);

		/// Manage the chain of \c spCore to use
		void			clear();
		bool			add(spCore* core);

		/// Access the results
		virtual int		getNOutputs() const;
		virtual const Tensor&	getOutput(int index) const;

		int			getNCores() const { return m_n_cores; }

	protected:

		/// Check if the input tensor has the right dimensions and type
		virtual bool		checkInput(const Tensor& input) const;

		/// Allocate (if needed) the output tensors given the input tensor dimensions
		virtual bool		allocateOutput(const Tensor& input);

		/// Process some input tensor (the input is checked, the outputs are allocated)
		virtual bool		processInput(const Tensor& input);

		//////////////////////////////////////////////////////////
		/// Attributes

		spCore**		m_cores;
		int			m_n_cores;
	};

/**
 * @}
 */

}

#endif
