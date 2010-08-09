#ifndef _TORCH5SPRO_MULTIVARIATE_DIAGONAL_GAUSSIAN_DISTRIBUTION_MACHINE_H_
#define _TORCH5SPRO_MULTIVARIATE_DIAGONAL_GAUSSIAN_DISTRIBUTION_MACHINE_H_

#include "MultiVariateNormalDistribution.h"
#include "Machines.h"

namespace Torch {

/** MultiVariateDiagonalGaussianDistribution

	@author Sebastien Marcel (marcel@idiap.ch)
*/
class MultiVariateDiagonalGaussianDistribution : public MultiVariateNormalDistribution
{
public:	
	///
	MultiVariateDiagonalGaussianDistribution();

	///
	MultiVariateDiagonalGaussianDistribution(int n_inputs_, int n_gaussians_);

	///
	~MultiVariateDiagonalGaussianDistribution();

	///
	virtual bool 		prepare();

	///
	virtual bool 		EMinit();

	///
	virtual bool 		EMupdate();
		
	//---
	
	///
	virtual bool 		resize(int n_inputs_, int n_means_);

	///
	virtual bool 		cleanup();

	///
	virtual bool 		sampleEMaccPosteriors(double *sample_, const double input_posterior);

	///
	virtual double 		sampleProbabilityOneGaussian(double *sample_, int g);

	///
	virtual double 		sampleProbability(double *sample_);

	///
	virtual bool		loadFile(File& file);

	///
	virtual bool		saveFile(File& file) const;

	/// Constructs an empty Machine of this kind - overriden
	// (used by <MachineManager>, this object should be deallocated by the user)
	virtual Machine*	getAnInstance() const { return new MultiVariateDiagonalGaussianDistribution(); }

	// Get the ID specific to each Machine - overriden
	virtual int		getID() const { return MULTIVARIATE_DIAGONAL_GAUSSIAN_DISTRIBUTION_MACHINE_ID; }

protected:
	//
	bool use_log;

	//
	double *posterior_numerator;
	double *g_norm;

};
	
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// REGISTER this machine to the <MachineManager>
const bool multivariate_diagonal_gaussian_machine_registered = MachineManager::getInstance().add(new MultiVariateDiagonalGaussianDistribution(), "MultiVariateDiagonalGaussianDistribution");
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
}

#endif


