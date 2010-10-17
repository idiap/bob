#ifndef _TORCH5SPRO_MACHINE_PCA_MACHINE_H_
#define _TORCH5SPRO_MACHINE_PCA_MACHINE_H_

#include "machine/EigenMachine.h"

namespace Torch {
/**
 * \ingroup libmachine_api
 * @{
 *
 */

/** This class is designed to handle Principal Component Analysis (PCA)
    @author Sebastien Marcel (marcel@idiap.ch)
    @author Laurent El Shafey (Laurent.El-Shafey@idiap.ch)
    @see SVDPCATrainer
*/
	class PCAMachine : public EigenMachine
	{
	public:
		/**
     * Constructor
     */
		PCAMachine();	

		/**
     * Constructor
     */
		PCAMachine(int n_inputs_);	

		/**
     * Destructor
     */
		virtual ~PCAMachine();


		/**
     * Initializes the machine
     */
		virtual bool	init_();

		/**
     * Computes the projection of the input onto the PCA matrix
     */
		virtual bool    forward(const Tensor& input);

		/**
     * Load the content from files (\emph{not the options}) - overriden
     */
		virtual bool    loadFile(File& file);

		/**
     * Save the content to files (\emph{not the options}) - overriden
     */
		virtual bool    saveFile(File& file) const;

  public:
	  /**
     *  Mean vector
     */
		double		*Xm;
	};

/**
 * @}
 */

	// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	// REGISTER this machine to the <MachineManager>
	const bool pca_machine_registered = MachineManager::getInstance().add(new PCAMachine(), "PCAMachine");
	// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

}

#endif

