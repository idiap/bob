#ifndef _TORCHVISION_LR_TRAINER_H_
#define _TORCHVISION_LR_TRAINER_H_

#include "Trainer.h"		// <LRTrainer> is a <Trainer>

namespace Torch
{
	class LRMachine;

	/////////////////////////////////////////////////////////////////////////
	// Torch::LRTrainer:
	//	- trains a LRMachine for two-class classification problems
	//	- it uses only on Nx1D DoubleTensors, trying to minimize
	//		the negative of the loglikelihood with L1 and L2 regularization terms
	//	- the targets must have values between 0.0 and 1.0!
	//	- uses a validation dataset (<setValidationData>) to set the optimum L1 and L2 norm priors
	//
        //      - PARAMETERS (name, type, default value, description):
        //		//
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class LRTrainer : public Torch::Trainer
	{
	public:

		// Constructor
		LRTrainer();

		// Destructor
		virtual ~LRTrainer();

		/// Train the given machine on the given dataset
		virtual bool 		train();

		// Set the validation dataset
		bool			setValidationData(DataSet* dataset);

		// Test the LR machine (returns the detection rate in percentages)
		static double		test(LRMachine* machine, DataSet* samples);

        private:

		/////////////////////////////////////////////////////////////////

		// Train the LR machine using the given L1 and L2 priors
		void			train(	double L1_prior, double L2_prior,
						double* weights, double* gradients, bool* fselected, int size,
						bool verbose);

		// Compute the gradient for the Grafting method (negative loglikelihoods + regularization terms)
		void			getGradient(double* gradients, const double* weights, int size,
							double L1_prior, double L2_prior);

                /////////////////////////////////////////////////////////////////
      		// Attributes

		DataSet*		m_validation_dataset;
	};
}

#endif
