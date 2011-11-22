/**
 * @file cxx/old/machine/machine/GradientMachine.h
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
#ifndef _TORCH5SPRO_GRADIENT_MACHINE_H_
#define _TORCH5SPRO_GRADIENT_MACHINE_H_

#include "machine/Machine.h"	// GradientMachine is a <Machine>

namespace Torch
{
        //////////////////////////////////////////////////////////////////////////////////////
	// Torch::GradientMachine:
	//
	// TODO: doxygen header!
	//////////////////////////////////////////////////////////////////////////////////////

	class GradientMachine : public Machine
	{
	public:
		/// Constructors
		GradientMachine();

		GradientMachine(const int n_inputs_, const int n_outputs_ = 0, const int n_parameters_ = 0);
		
		/// Destructor
		virtual ~GradientMachine();

		///
		virtual bool 		resize(const int n_inputs_, const int n_outputs_, const int n_parameters_ = 0);

		/// !!! rename into prepare !!!
		virtual bool 		prepare();

		/// !!! rename into shuffle !!!
		virtual bool 		shuffle() { return true; };

		/// !!! rename into Ginit !!! -- Gradient Init
		virtual bool 		Ginit() { return true; };
		
		/// !!! rename into Gupdate !!! -- Gradient update
		virtual bool 		Gupdate(double learning_rate) { return true; };
		
		///////////////////////////////////////////////////////////

		///
		virtual bool 		forward(const Tensor& input);
		virtual bool 		forward(const DoubleTensor *input) = 0;

		///
		virtual bool 		backward(const Tensor& input, const DoubleTensor *alpha);
		virtual bool 		backward(const DoubleTensor *input, const DoubleTensor *alpha) = 0;

		// Loading/Saving the content from files (<em>not the options</em>) - overriden
		virtual bool		loadFile(File& file);
		virtual bool		saveFile(File& file) const;

		///////////////////////////////////////////////////////////

		int getNinputs() { return n_inputs; };
		int getNoutputs() { return n_outputs; };

		DoubleTensor*	m_beta; // the derivative with respect to the inputs

	protected:
		int n_inputs;
		int n_outputs;
		int n_parameters;
		double *parameters;
		double *der_parameters;
	};

}

#endif
