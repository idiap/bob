#ifndef MEMORY_DATA_SET_INC
#define MEMORY_DATA_SET_INC

#include "DataSet.h"

namespace Torch 
{
	class MemoryDataSet : public DataSet
	{
	public:
		///
		MemoryDataSet();

		///
		MemoryDataSet(int n_examples, bool has_targets = false);

		/// 
		virtual TensorPair &operator()(long) const;

		///
		virtual ~MemoryDataSet();

	protected:

		///
		void cleanup();

	private:
		///
		TensorPair *examples;
	};

}

#endif
