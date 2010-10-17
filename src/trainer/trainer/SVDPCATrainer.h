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

