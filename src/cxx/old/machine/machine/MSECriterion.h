/**
 * @file cxx/old/machine/machine/MSECriterion.h
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
#ifndef _TORCH5SPRO_MSE_CRITERION_H_
#define _TORCH5SPRO_MSE_CRITERION_H_

#include "machine/Criterion.h"

namespace Torch
{
        //////////////////////////////////////////////////////////////////////////////////////
	// Torch::MSECriterion:
	//
	// TODO: doxygen header!
	//////////////////////////////////////////////////////////////////////////////////////

	class MSECriterion : public Criterion
	{
	public:

		/// Constructor
		MSECriterion(const int target_size);
		
		/// Destructor
		virtual ~MSECriterion();

		///////////////////////////////////////////////////////////

		///
		virtual bool 	forward(const DoubleTensor *machine_output, const Tensor *target);

		///////////////////////////////////////////////////////////
	};

}

#endif
