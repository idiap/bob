#ifndef MEMORY_DATA_SET_INC
#define MEMORY_DATA_SET_INC

#include "DataSet.h"

namespace Torch 
{
	class MemoryDataSet : public DataSet
	{
	public:
		///
		MemoryDataSet(Tensor::Type example_type_ = Tensor::Double, Tensor::Type target_type_ = Tensor::Short);

		///
		MemoryDataSet(int n_examples_, Tensor::Type example_type_ = Tensor::Double, bool has_targets = false, Tensor::Type target_type_ = Tensor::Short);

		///
		virtual Tensor* getExample(long);
		virtual Tensor &operator()(long);

		///
		virtual Tensor* getTarget(long);

		///
		virtual ~MemoryDataSet();

	protected:

		///
		void cleanup();

	private:
		///
		Tensor **examples;
		Tensor **targets;
	};

}

#endif
