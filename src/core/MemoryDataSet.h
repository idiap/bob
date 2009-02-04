#ifndef MEMORY_DATA_SET_INC
#define MEMORY_DATA_SET_INC

#include "DataSet.h"

namespace Torch
{
	class MemoryDataSet : public DataSet
	{
	public:

		// Constructor
		MemoryDataSet(	int n_examples,
				Tensor::Type example_type_ = Tensor::Double,
				Tensor::Type target_type_ = Tensor::Short);

		// Destructor
		virtual ~MemoryDataSet();

		// Reinitialize the dataset
		void		reset(	int n_examples,
					Tensor::Type example_type_ = Tensor::Double,
					Tensor::Type target_type_ = Tensor::Short);

		// Access examples
		virtual Tensor* getExample(long index);
		virtual Tensor&	operator()(long index);

		// Access targets
		virtual Tensor* getTarget(long index) = 0;
		virtual void	setTarget(long index, Tensor* target) = 0;

	private:

		/// Delete the allocated tensors
		void 		cleanup();

		/// Allocated examples and targets
		Tensor**	m_examples;	// Array of allocated tensors
		Tensor**	m_targets;	// Array of pointers to external tensors
	};

}

#endif
