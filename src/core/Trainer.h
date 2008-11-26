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
		Trainer(Machine& machine, DataSet& dataset);

		/// Destructor
		virtual ~Trainer();

		/// Train the machine on the given dataset
		virtual bool 		train() = 0;

	protected:

		////////////////////////////////////////////////////
		/// Attributes

		Machine&		m_machine;	// The machine that will be trained
		DataSet&		m_dataset;	// The dataset used to train the machine
	};

}

#endif
