/**
 * @file cxx/old/machine/machine/IntLutMachine.h
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
#ifndef _TORCH5SPRO_INTLUT_MACHINE_H_
#define _TORCH5SPRO_INTLUT_MACHINE_H_

#include "machine/Machine.h"
#include "machine/Machines.h"

namespace Torch {


	//////////////////////////////////////////////////////////////////////////////////////
	// Torch::LUTMachine:
	//      Process some input using a model (loaded from some file).
	//      The output is a DoubleTensor!
	//
	//      NB: The ouput should be allocated and deallocated by each Machine implementation!
	//
	//	EACH MACHINE SHOULD REGISTER
	//		==> MachineManager::GetInstance().add(new XXXMachine) <==
	//	TO THE MACHINEMANAGER CLASS!!!
	//
	// TODO: doxygen header!
	//////////////////////////////////////////////////////////////////////////////////////

	class IntLutMachine : public Machine
	{

	public:

		/// Constructor
		IntLutMachine();

		/// Destructor
		virtual ~IntLutMachine();

		/// Process the input tensor
		virtual bool 		forward(const Tensor& input);

		// Loading/Saving the content from files (<em>not the options</em>) - overriden
		virtual bool		loadFile(File& file);
		virtual bool		saveFile(File& file) const;

		/// Constructs an empty Machine of this kind
		// (used by <MachineManager>, this object is automatically deallocated)
		virtual Machine*	getAnInstance() const { return manage(new IntLutMachine()); }

		// Get the ID specific to each Machine
		virtual int		getID() const { return INT_LUT_MACHINE_ID; }

		void 			setParams(int n_bins_, double *lut_);

		///////////////////////////////////////////////////////////
		// Access functions

		int			getLUTSize() const { return n_bins; }
		double*			getLUT()  { return lut; }

		///////////////////////////////////////////////////////////

	protected:

		///////////////////////////////////////////////////////////////
		// Attributes

		// parameters of the machine
		//double min, max;
		int n_bins;
		double *lut;
	};


	// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // REGISTER this machine to the <MachineManager>
        const bool intlut_machine_registered = MachineManager::getInstance().add(
		new IntLutMachine(), "IntLutMachine");
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
}

#endif
