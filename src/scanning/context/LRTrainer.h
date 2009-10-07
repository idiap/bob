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
	//	- the targets must have values between 0.0 and 1.0, better between 0.1 and 0.9!
	//	- uses a validation dataset (<setValidationData>) to set the optimum L1 and L2 norm priors
	//
        //      - PARAMETERS (name, type, default value, description):
        //		"FARvsFRRRatio"		double		1.0	"FAR vs FRR ratio: between 0.0 and 1.0"
        //		"useL1"			bool		false	"Use L1 norm regularization term"
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

		// Test the LR machine
		static void		test(	LRMachine* machine, DataSet* samples,
						double& TAR, double& FAR, double& HTER,
						double FARvsFRRRatio);

		// Compute the gradient for the Grafting method (negative loglikelihoods + regularization terms)
		static void		getGradient(	DataSet* dataset,
							double* gradients, double* buf_gradients,
							const double* weights, int size,
							double L1_prior, double L2_prior, double FARvsFRRRatio);

		// Computes the inverse of the number of positive and negative samples in a dataset
		static void		getInvPosNeg(	DataSet* dataset,
							double& inv_n_pos, double& inv_n_neg);

        private:

		/////////////////////////////////////////////////////////////////

		// Train the LR machine using the given L1 and L2 priors
		bool			train(	double L1_prior, double L2_prior, double FARvsFRRRatio,
						double* weights, double* gradients, double* buf_gradients,
						bool* fselected, int size,
						bool verbose);

		// Optimize the LR machine - the optimum threshold is set to the machine!
		static void		optimize(LRMachine* machine, DataSet* samples, double FARvsFRRRatio);

                /////////////////////////////////////////////////////////////////
      		// Attributes

		DataSet*		m_validation_dataset;
	};
}

#endif
