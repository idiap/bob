#ifndef DATA_SET_INC
#define DATA_SET_INC

#include "Object.h"
#include "Tensor.h"
//#include "PreProcessing.h"

namespace Torch 
{
	class DataSet : public Object
	{
	public:
		///
		DataSet(Tensor::Type example_type_ = Tensor::Double, Tensor::Type target_type_ = Tensor::Short);

		///
		//virtual void preProcess(PreProcessing *pre_processing) = 0;

		///
		virtual Tensor* getExample(long) = 0;
		virtual Tensor &operator()(long) = 0;

		///
		virtual Tensor* getTarget(long) = 0;

		/// Query the number of examples in the dataset
		int getNumberOfExamples() const { return n_examples; };

		///
		virtual ~DataSet();
	
	protected:

		/// Number of examples in the dataset.
		int n_examples;

		Tensor::Type example_type;
		Tensor::Type target_type;
    
	};

}

#endif
