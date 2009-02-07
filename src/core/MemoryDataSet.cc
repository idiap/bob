#include "Tensor.h"
#include "MemoryDataSet.h"

namespace Torch {

	//////////////////////////////////////////////////////////////////////////////
	// \breif Constructor 
	//
	MemoryDataSet::MemoryDataSet(int n_examples_, Tensor::Type example_type_) 
		: DataSet(example_type_)
	{
		// indicate that the dataset is empty so far
		m_examples = NULL;
		m_targets  = NULL;

		// allocated the data set by using reset
		reset(n_examples_, example_type_);

	}

	//////////////////////////////////////////////////////////////////////////////
	// \breif getExample

	MemoryDataSet::~MemoryDataSet() 
	{
		// free all allocated memory
		cleanup();
	}

	//////////////////////////////////////////////////////////////////////////////
	// \breif getExample
	// \param[in] 	index  	Which element do you want a pointer to  
	// \param[out] 	Tensor* Pointer to example

	Tensor* MemoryDataSet::getExample(long index)
	{
		if(m_examples == NULL) 
			error("MemoryDataSet(): no examples in memory.");

		// make sure that the index is in range
		if(!isInRange(index)) 
			error("MemoryDataSet(): example (%d) out-of-range [0-%d].", index, m_size - 1);

		return m_examples[index];
	}

	//////////////////////////////////////////////////////////////////////////////
	// \breif isInRange 
	// \param[in] 	index	Does the index make sence
	// \param[out]	bool	True or False

	bool MemoryDataSet::isInRange(long index) 
	{
		return (index >= 0 && index < m_size) ? true : false; 
	}

	//////////////////////////////////////////////////////////////////////////////
	// \breif getTarget 

	Tensor* MemoryDataSet::getTarget(long index)
	{
		if(m_targets == NULL) 
			error("MemoryDataSet(): no examples in memory.");

		if(!isInRange(index)) 
			error("MemoryDataSet(): target (%d) out-of-range [0-%d].", index, m_size - 1);

		return m_targets[index];
	}
	
	//////////////////////////////////////////////////////////////////////////////
	// \breif getTarget 

	void MemoryDataSet::setTarget(long index, Tensor* target)
	{
		if(m_targets == NULL) 
			error("MemoryDataSet(): no examples in memory.");

		if(!isInRange(index)) 
			error("MemoryDataSet(): target (%d) out-of-range [0-%d].", index, m_size - 1);

		m_targets[index] = target;
	}

	//////////////////////////////////////////////////////////////////////////////
	// \breif getTarget 

	void MemoryDataSet::reset(int n_examples, Tensor::Type example_type_ )
	{
		// clean up old
		cleanup();

		// register the size of the DataSet
		m_size = n_examples;

		// create the two main arrays of pointers.
		m_examples 	= new Tensor* [m_size];
		m_targets	= new Tensor* [m_size];

		// swith over typ of tensor and size of tensor
		switch(example_type_) {

			case Tensor::Char:

				// allocate a ton of Char tensors
				for (int cnt = 0; cnt < m_size; ++cnt) {
					m_examples[cnt] = new CharTensor;
				}
				break;

			case Tensor::Short:

				// allocate a ton of Short tensors
				for (int cnt = 0; cnt < m_size; ++cnt) {
					m_examples[cnt] = new ShortTensor;
				}
				break;

			case Tensor::Int:

				// allocate a ton of Int tensors
				for (int cnt = 0; cnt < m_size; ++cnt) {
					m_examples[cnt] = new IntTensor;
				}
				break;

			case Tensor::Long:

				// allocate a ton of Long tensors
				for (int cnt = 0; cnt < m_size; ++cnt) {
					m_examples[cnt] = new LongTensor;
				}
				break;


			case Tensor::Float:

				// allocate a ton of Float tensors
				for (int cnt = 0; cnt < m_size; ++cnt) {
					m_examples[cnt] = new FloatTensor;
				}
				break;

			case Tensor::Double:

				// allocate a ton of Double tensors
				for (int cnt = 0; cnt < m_size; ++cnt) {
					m_examples[cnt] = new DoubleTensor;
				}
				break;


			default:
				// error
				error("MemoryDataSet: sorry target type not supported yet");

				for (int cnt = 0; cnt < m_size; ++cnt) {
					m_examples[cnt] = NULL;
				}
		}

		// set all the target pointers to NULL
		for (int cnt = 0; cnt < m_size; ++cnt) {
			m_targets[cnt]  = NULL;
		}
	}


	//////////////////////////////////////////////////////////////////////////////
	// \breif getTarget 

	void MemoryDataSet::cleanup()
	{
		if (NULL != m_examples) {

			// remove all the examples
			for(int i = 0 ; i < m_size ; i++) 
				delete m_examples[i];

			// free 
			delete [] m_examples;
		}

		if (NULL != m_targets) {

			// remove all the targets
			for(int i = 0 ; i < m_size ; i++) 
				delete m_targets[i];

			// free
			delete [] m_targets;
		}
	}

	//////////////////////////////////////////////////////////////////////////////
	// \breif getTarget 

	Tensor &MemoryDataSet::operator()(long index)
	{
		if(m_examples == NULL) 
			error("MemoryDataSet(): no examples in memory.");

		if(!isInRange(index))
			error("MemoryDataSet(): example (%d) out-of-range [0-%d].", index , m_size - 1);

		return *(m_examples[index]);
	}
}
