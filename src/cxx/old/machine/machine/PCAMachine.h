/**
 * @file cxx/old/machine/machine/PCAMachine.h
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

