#ifndef DATA_SET_INC
#define DATA_SET_INC

#include "Object.h"
//#include "PreProcessing.h"

namespace Torch 
{
   	class Tensor;

   	struct TensorSize
	{
		TensorSize()
		{
			n_dimensions = 0;
			size[0] = 0;
			size[1] = 0;
			size[2] = 0;
			size[3] = 0;
		}
		TensorSize(long dim0)
		{
			n_dimensions = 1;
			size[0] = dim0;
			size[1] = 0;
			size[2] = 0;
			size[3] = 0;
		}

		TensorSize(long dim0, long dim1)
		{
			n_dimensions = 2;
			size[0] = dim0;
			size[1] = dim1;
			size[2] = 0;
			size[3] = 0;
		}

		TensorSize(long dim0, long dim1, long dim2)
		{
			n_dimensions = 3;
			size[0] = dim0;
			size[1] = dim1;
			size[2] = dim2;
			size[3] = 0;
		}

		TensorSize(long dim0, long dim1, long dim2, long dim3)
		{
			n_dimensions = 4;
			size[0] = dim0;
			size[1] = dim1;
			size[2] = dim2;
			size[3] = dim3;
		}

		int n_dimensions;
		long size[4];
	};

   	struct TensorPair
	{
	   	TensorPair() { input = NULL; target = NULL; };

		Tensor *input;
		Tensor *target;
	};

	class DataSet : public Object
	{
	public:
		///
		DataSet();

		///
		//virtual void preProcess(PreProcessing *pre_processing) = 0;

		/// Query an example i.e a pair of Tensors (example, target)
		virtual TensorPair &operator()(long) const = 0;

		/// Query the number of examples in the dataset
		int getNumberOfExamples() const { return n_examples; };

		///
		virtual ~DataSet();
	
	protected:

		/// Number of examples in the dataset.
		int n_examples;
    
	};

}

#endif
