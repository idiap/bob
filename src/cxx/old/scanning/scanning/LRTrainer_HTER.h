/**
 * @file cxx/old/scanning/scanning/LRTrainer_HTER.h
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
#ifndef _TORCHVISION_LR_TRAINER_H_
#define _TORCHVISION_LR_TRAINER_H_

#include "trainer/Trainer.h"		// <LRTrainer> is a <Trainer>

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

		// Test the LR machine
		static void		test(	LRMachine* machine, DataSet* samples,
						double& TAR, double& FAR, double& HTER);

		// Compute the gradient for the Grafting method (negative loglikelihoods + regularization terms)
		static void		getGradient(	DataSet* dataset,
							double* gradients, double* buf_gradients,
							const double* weights, int size,
							double L1_prior, double L2_prior);

		// Computes the inverse of the number of positive and negative samples in a dataset
		static void		getInvPosNeg(	DataSet* dataset,
							double& inv_n_pos, double& inv_n_neg);

        private:

		/////////////////////////////////////////////////////////////////

		// Train the LR machine using the given L1 and L2 priors
		bool			train(	double L1_prior, double L2_prior,
						double* weights,
						double* gradients, double* buf_gradients,
						bool* fselected, int size,
						bool verbose);

		// Optimize (minimum HTER) and test the LR machine
		// NB: the optimum threshold is set to the machine!
		static void		optimize(LRMachine* machine, DataSet* samples);

                /////////////////////////////////////////////////////////////////
      		// Attributes

		DataSet*		m_validation_dataset;
	};
}

#endif
