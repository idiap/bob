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
		DataSet( Tensor::Type example_type_ = Tensor::Double );

		// Destructor
		virtual ~DataSet();

		// Access examples
		virtual Tensor* getExample(long) = 0;
		virtual Tensor&	operator()(long) = 0;

		// Access targets
		virtual Tensor* getTarget(long) = 0;
		virtual void	setTarget(long, Tensor*) = 0;

		/// Query the number of examples in the dataset
		int 		getNumberOfExamples() const { return m_size; };


	protected:

		// Number of examples in the dataset.
		long 		m_size;

		Tensor::Type 	m_example_type;
		Tensor::Type 	m_target_type;

	private:
		bool isInRange(long index);

	};

}

#endif
