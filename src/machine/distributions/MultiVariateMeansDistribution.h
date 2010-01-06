#ifndef _TORCH5SPRO_MULTIVARIATE_MEANS_DISTRIBUTION_MACHINE_H_
#define _TORCH5SPRO_MULTIVARIATE_MEANS_DISTRIBUTION_MACHINE_H_

#include "MultiVariateNormalDistribution.h"
#include "Machines.h"

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
	virtual bool 		sampleEMaccPosteriors(double *sample_, const double input_posterior);

	///
	virtual double 		sampleProbabilityOneMean(double *sample_, int m);

	///
	virtual double 		sampleProbability(double *sample_);

	///
	virtual bool		loadFile(File& file);

	///
	virtual bool		saveFile(File& file) const;

	/// Constructs an empty Machine of this kind - overriden
	/// (used by <MachineManager>, this object should be deallocated by the user)
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


