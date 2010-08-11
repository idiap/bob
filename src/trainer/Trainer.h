#ifndef TRAINER_INC
#define TRAINER_INC

#include "core/Object.h"
#include "core/DataSet.h"
#include "machine/Machine.h"

/**
 * \addtogroup libtrainer_api libTrainer API
 * @{
 *
 *  The libTrainer API.
 */
namespace Torch {

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

}

/**
 * @}
 */


/**
@page libTrainer Trainer: a Trainer module

@section intro Introduction

Trainer is module of Torch used to train a Machine.

@section api Documentation
- @ref libtrainer_api "libTrainer API"

*/

#endif
