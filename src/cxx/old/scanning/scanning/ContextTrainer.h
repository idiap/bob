/**
 * @file cxx/old/scanning/scanning/ContextTrainer.h
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
#ifndef _TORCHVISION_SCANNING_CONTEXT_TRAINER_H_
#define _TORCHVISION_SCANNING_CONTEXT_TRAINER_H_

#include "trainer/Trainer.h"		// <ContextTrainer> is a <Trainer>

namespace Torch
{
	class ContextMachine;
	class ContextDataSet;

	/////////////////////////////////////////////////////////////////////////
	// Torch::ContextTrainer:
	//	- trains ContextMachine using two ContextDataSets:
	//		- one for training and one for validation
        //
        //      - PARAMETERS (name, type, default value, description):
        //		//
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class ContextTrainer : public Torch::Trainer
	{
	public:

		// Constructor
		ContextTrainer();

		// Destructor
		virtual ~ContextTrainer();

		// Train the given machine on the given dataset
		virtual bool 		train();

		// Set the validation dataset
		bool			setValidationData(DataSet* dataset);

		// Test the Context machine
		static void		test(ContextMachine* machine, ContextDataSet* samples,
						double& TAR, double& FAR, double& HTER);

        private:

                /////////////////////////////////////////////////////////////////
      		// Attributes

		DataSet*		m_validation_dataset;
	};
}

#endif
