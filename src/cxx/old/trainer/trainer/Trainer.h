/**
 * @file cxx/old/trainer/trainer/Trainer.h
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
#ifndef TRAINER_INC
#define TRAINER_INC

#include "core/Object.h"
#include "core/DataSet.h"
#include "machine/Machine.h"

namespace Torch
{

/**
 * \defgroup libtrainer_api libTrainer API
 * @{
 *
 *  The libTrainer API.
 */

	class Trainer : public Object
	{
	public:

		/// Constructor
		Trainer();

		/// Destructor
		virtual ~Trainer();

		/// Train the given machine on the given dataset
		virtual bool 	train() = 0;

		/// Set the DataSet to train with
		bool 		setData(DataSet *m_dataset_);

		/// Set the Machine to train
		bool 		setMachine(Machine *m_machine_);

		///
		Machine* 	getMachine() const { return m_machine; } 

	protected:
		////////////////////////////////////////////////////
		/// Attributes

		Machine		*m_machine;	// The machine that will be trained
		DataSet		*m_dataset;	// The dataset used to train the machine
	};

/**
 * @}
 */

}


/**
@page libTrainer Trainer: a Trainer module

@section intro Introduction

Trainer is module of Torch used to train a Machine.

@section api Documentation
- @ref libtrainer_api "libTrainer API"

*/

#endif
