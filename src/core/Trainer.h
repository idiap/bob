#ifndef TRAINER_INC
#define TRAINER_INC

#include "Object.h"

namespace Torch {

	class Machine;
	class DataSet;

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

	protected:
		////////////////////////////////////////////////////
		/// Attributes

		Machine		*m_machine;	// The machine that will be trained
		DataSet		*m_dataset;	// The dataset used to train the machine
	};

}

#endif
