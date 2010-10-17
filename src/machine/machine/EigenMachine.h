#ifndef _TORCH5SPRO_MACHINE_EIGEN_MACHINE_H_
#define _TORCH5SPRO_MACHINE_EIGEN_MACHINE_H_

#include "machine/Machines.h"
#include "machine/Machine.h"
#include "core/DataSet.h"

namespace Torch
{
/**
 * \ingroup libmachine_api
 * @{
 *
 */

  /**
   * This class defines an Eigenmachine.
   * An Eigenmachine performs dimensionality reduction by projecting data on a basis of Eigenvectors.
   * Both PCAMachine and LDAMachine are Eigenmachines.
   */
	class EigenMachine : public Machine
	{
	public:

		/**
     * Constructor
     */
		EigenMachine();

		/**
     * Constructor
     */
		EigenMachine(const int n_inputs_ );

		/**
     * Destructor
     */
		virtual ~EigenMachine();


		/**
     * Resize the output vector
     */
		virtual bool 	resize(const int n_outputs_);
		virtual bool  resize(const int n_outputs_, const int n_frames_per_sequence_);
		virtual bool  resize(const int n_outputs_, const int n_frames_per_sequence_, const int n_sequences_per_sequence_);

    /**
     * Initialize the Eigenmachine
     */
		virtual bool 	init_();

    /**
     * Determine the number of Eigenvectors to keep
     */
		virtual bool 	setNumberOfRelevantEigenvectors();

		/**
     * Process an Eigenvectors
     */
		virtual bool 	forward(const Tensor& input);

		/**
      * Load the Eigenvectors from a file
      */
		virtual bool	loadFile(File& file);

		/**
      * Save the Eigenvectors from a file
      */
		virtual bool	saveFile(File& file) const;

		/**
     *  Constructs an empty Machine of this kind - overriden
		 * (used by <MachineManager>, this object should be deallocated by the user)
		 */
		virtual Machine*	getAnInstance() const { return new EigenMachine(); }

		/**
     * Get the ID specific to each Machine - overriden
     */
		virtual int		getID() const { return EIGEN_MACHINE_ID; }


	public:
		int 		  n_inputs;
		int		    n_outputs;

		double		*eigenvalues;
		double		*eigenvectors;

		double		variance;

    DoubleTensor 	*frame_in_;
    DoubleTensor 	*sequence_in_;
    DoubleTensor 	*frame_out_;
    DoubleTensor 	*sequence_out_;
	};

/**
 * @}
 */

  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  // REGISTER this machine to the <MachineManager>
  const bool eigen_machine_registered = MachineManager::getInstance().add(new EigenMachine(), "EigenMachine");
  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
}

#endif

