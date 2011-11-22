/**
 * @file cxx/old/machine/machine/MultiVariateMeansDistribution.h
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
#ifndef _TORCH5SPRO_MULTIVARIATE_MEANS_DISTRIBUTION_MACHINE_H_
#define _TORCH5SPRO_MULTIVARIATE_MEANS_DISTRIBUTION_MACHINE_H_

#include "machine/MultiVariateNormalDistribution.h"
#include "machine/Machines.h"

namespace Torch {

/** MultiVariateMeansDistribution

	@author Sebastien Marcel (marcel@idiap.ch)
*/
class MultiVariateMeansDistribution : public MultiVariateNormalDistribution
{
public:	
	///
	MultiVariateMeansDistribution();

	///
	MultiVariateMeansDistribution(int n_inputs_, int n_means_);

	///
	~MultiVariateMeansDistribution();

	///
	virtual bool 		EMupdate();
	
	//---
	
	///
	virtual bool 		sampleEMaccPosteriors(const DoubleTensor& sample_, const double input_posterior);

	///
	virtual double 		sampleProbabilityOneMean(const DoubleTensor& sample_, int m);

	///
	virtual double 		sampleProbability(const DoubleTensor& sample_);

	///
	virtual bool		loadFile(File& file);

	///
	virtual bool		saveFile(File& file) const;

	/// Constructs an empty Machine of this kind - overriden
	// (used by <MachineManager>, this object should be deallocated by the user)
	virtual Machine*	getAnInstance() const { return new MultiVariateMeansDistribution(); }

	// Get the ID specific to each Machine - overriden
	virtual int		getID() const { return MULTIVARIATE_MEANS_DISTRIBUTION_MACHINE_ID; }
};
	
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// REGISTER this machine to the <MachineManager>
const bool multivariate_means_machine_registered = MachineManager::getInstance().add(new MultiVariateMeansDistribution(), "MultiVariateMeansDistribution");
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
}

#endif


