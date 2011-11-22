/**
 * @file cxx/old/machine/machine/ProbabilityDistribution.h
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
#ifndef _TORCH5SPRO_PROBABILITY_DISTRIBUTION_MACHINE_H_
#define _TORCH5SPRO_PROBABILITY_DISTRIBUTION_MACHINE_H_

#include "machine/Machine.h"	// ProbabilityDistribution is a <Machine>

namespace Torch
{
        //////////////////////////////////////////////////////////////////////////////////////
	// Torch::ProbabilityDistribution:
	//
	// TODO: doxygen header!
	//////////////////////////////////////////////////////////////////////////////////////

	class ProbabilityDistribution : public Machine
	{
	public:
		/// Constructor
		ProbabilityDistribution();

		/// Constructor
		ProbabilityDistribution(const int n_inputs_);
		
		/// Destructor
		virtual ~ProbabilityDistribution();

		///
		virtual bool 		prepare() { return true; };
		
		///
		virtual bool 		EMinit() { return true; };

		///
		virtual bool 		EMaccPosteriors(const DoubleTensor& input, const double input_posterior) { return true; };

		///
		virtual bool 		EMupdate() { return true; };
		
		///
		virtual bool 		forward(const Tensor& input);

		///
		virtual bool 		forward(const DoubleTensor *input) = 0;

		///
		virtual bool 		print() { return true; };
		
		///
		virtual bool 		shuffle() { return true; };

		///
		int			getNinputs() { return n_inputs; };
		
	protected:
		int n_inputs;
	};

}

#endif
