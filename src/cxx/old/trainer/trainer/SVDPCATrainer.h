/**
 * @file cxx/old/trainer/trainer/SVDPCATrainer.h
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
#ifndef _TORCH5SPRO_TRAINER_SVDPCA_TRAINER_H_
#define _TORCH5SPRO_TRAINER_SVDPCA_TRAINER_H_

#include "trainer/Trainer.h"
#include "trainer/EigenTrainer.h"
#include "machine/PCAMachine.h"

namespace Torch {

/** This class is designed to compute Principal Component Analysis
    @author Sebastien Marcel (marcel@idiap.ch)
    @author Laurent El Shafey (Laurent.El-Shafey@idiap.ch)
*/
	class SVDPCATrainer : public Trainer
	{
	public:
/*   		/// number of inputs
		int n_inputs;

   		/// covariance matrix of the data
		DoubleTensor *CovMat;
   	
		/// mean vector
		DoubleTensor *Xm_;

		/// eigenvectors
	    	DoubleTensor *eigenvectors;

		/// eigenvectors of the PCAMachine (to be filled)
		DoubleTensor *eigenvectors_;

		/// eigenvalues
	    	DoubleTensor *eigenvalues;

		/// eigenvalues of the PCAMachine (to be filled)
		DoubleTensor *eigenvalues_;

		/// save the covariance matrix
		bool saveCovar;
*/	
		//-----
	
		/**
     * Constructor
     */
		SVDPCATrainer();

		/**
     * Launch the training process
     */
		virtual bool train();

		/**
     * Destructor
     */
		virtual ~SVDPCATrainer();
	};

}

#endif

