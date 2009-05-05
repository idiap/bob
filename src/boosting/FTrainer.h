#ifndef F_TRAINER_INC
#define F_TRAINER_INC

#include "Trainer.h"


namespace Torch {

	class FTrainer : public Trainer
	{
	public:

		/// Constructor
		FTrainer();

		/// Destructor
		virtual ~FTrainer();

		/// Train the given machine on the given dataset
	//	virtual bool 	train() = 0;
        // get the  output of trained trainer
		virtual double forward(Tensor *example_)=0; //get the output score for a single stage


	};

}

#endif
