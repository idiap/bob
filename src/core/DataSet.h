#ifndef DATA_SET_INC
#define DATA_SET_INC

#include "Object.h"
#include "Tensor.h"

namespace Torch
{
	class DataSet : public Object
	{
	public:
		// Constructor
		DataSet(Tensor::Type example_type_ = Tensor::Double, bool has_targets_ = false, Tensor::Type target_type_ = Tensor::Short);

		// Destructor
		virtual ~DataSet();

		// Access examples
		virtual Tensor* getExample(long) = 0;
		virtual Tensor&	operator()(long) = 0;

		// Access targets
		virtual Tensor* getTarget(long) = 0;
		virtual void	setTarget(long, Tensor*) = 0;

		/// Query the number of examples in the dataset
		int 		getNoExamples() const { return m_n_examples; };

		/// Query if the DataSet has targets
		bool 		hasTargets() const { return m_has_targets; };

	protected:
		//
		bool m_has_targets;

		// Number of examples in the dataset.
		long 		m_n_examples;

		// Type of the examples
		Tensor::Type 	m_example_type;

		// Type of the targets
		Tensor::Type 	m_target_type;
	};

}

#endif
