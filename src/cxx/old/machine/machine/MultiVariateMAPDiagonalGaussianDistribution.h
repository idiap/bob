/**
 * @file cxx/old/machine/machine/MultiVariateMAPDiagonalGaussianDistribution.h
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
#ifndef _TORCH5SPRO_MULTIVARIATE_MAP_DIAGONAL_GAUSSIAN_DISTRIBUTION_MACHINE_H_
#define _TORCH5SPRO_MULTIVARIATE_MAP_DIAGONAL_GAUSSIAN_DISTRIBUTION_MACHINE_H_

#include "machine/MultiVariateDiagonalGaussianDistribution.h"
#include "machine/Machines.h"
#include <boost/scoped_array.hpp>

namespace Torch {

/** MultiVariateMAPDiagonalGaussianDistribution

	@author Sebastien Marcel (marcel@idiap.ch)
*/
class MultiVariateMAPDiagonalGaussianDistribution : public MultiVariateDiagonalGaussianDistribution
{
public:	
	///
	MultiVariateMAPDiagonalGaussianDistribution(MultiVariateDiagonalGaussianDistribution *prior_);

	///
	~MultiVariateMAPDiagonalGaussianDistribution();

	///
	virtual bool 		prepare();

	///
	virtual bool 		sampleEMaccPosteriors(const DoubleTensor& sample_, const double input_posterior);

	///
	virtual bool 		EMupdate();
		
protected:
	MultiVariateDiagonalGaussianDistribution *prior;
	boost::scoped_array<double*> prior_means;

	float map_factor;
};
	
}

#endif


