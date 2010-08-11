#ifndef MEMORY_DATA_SET_INC
#define MEMORY_DATA_SET_INC

#include "core/DataSet.h"
#include "core/Tensor.h"

namespace Torch
{
	class MemoryDataSet : public DataSet
	{
	public:

		// Constructor
		MemoryDataSet(int n_examples = 0, Tensor::Type example_type_ = Tensor::Double, bool has_targets_ = false, Tensor::Type target_type_ = Tensor::Short);

		// Destructor
		virtual ~MemoryDataSet();

		// Reinitialize the dataset
		void		reset(int n_examples, Tensor::Type example_type_ = Tensor::Double, bool has_targets_ = false, Tensor::Type target_type_ = Tensor::Short);

		// Access examples
		virtual Tensor* getExample(long index);
		virtual Tensor&	operator()(long index);

		// Access targets
		virtual Tensor* getTarget(long index);
		virtual void	setTarget(long index, Tensor* target);

	private:

		/// Delete the allocated tensors
		void 		cleanup();

		/// Allocated examples and targets
		Tensor**	m_examples;	// Array of allocated tensors
		Tensor**	m_targets;	// Array of pointers to external tensors
	};

}

#endif
