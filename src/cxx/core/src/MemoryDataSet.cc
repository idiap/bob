#include "core/array_common.h"
#include "core/Tensor.h"
#include "core/MemoryDataSet.h"

namespace Torch {

	//////////////////////////////////////////////////////////////////////////////
	// \breif Constructor
	//
	MemoryDataSet::MemoryDataSet(int n_examples_, Tensor::Type example_type_, bool has_targets_, Tensor::Type target_type_)
		: DataSet(example_type_, has_targets_, target_type_)
	{
		// indicate that the dataset is empty so far
		m_examples = NULL;
		m_targets  = NULL;

		// allocated the data set by using reset
		reset(n_examples_, example_type_, has_targets_, target_type_);
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
		if(!isIndex(index, m_n_examples))
			error("MemoryDataSet(): example (%d) out-of-range [0-%d].", index, m_n_examples - 1);

		return m_examples[index];
	}

	Tensor& MemoryDataSet::operator()(long index)
	{
		return *getExample(index);
	}

	//////////////////////////////////////////////////////////////////////////////
	// \breif getTarget

	Tensor* MemoryDataSet::getTarget(long index)
	{
		if(m_targets == NULL)
			error("MemoryDataSet(): no targets in memory.");

		if(!isIndex(index, m_n_examples))
			error("MemoryDataSet(): target (%d) out-of-range [0-%d].", index, m_n_examples - 1);

		return m_targets[index];
	}

	//////////////////////////////////////////////////////////////////////////////
	// \breif getTarget

	void MemoryDataSet::setTarget(long index, Tensor* target)
	{
		if(m_targets == NULL)
			error("MemoryDataSet(): no targets in memory.");

		if(!isIndex(index, m_n_examples))
			error("MemoryDataSet(): target (%d) out-of-range [0-%d].", index, m_n_examples - 1);

		m_targets[index] = target;
	}

	//////////////////////////////////////////////////////////////////////////////
	// \breif getTarget

	void MemoryDataSet::reset(int n_examples, Tensor::Type example_type_, bool has_targets_, Tensor::Type target_type_)
	{
		// clean up old
		cleanup();

		m_example_type = example_type_;
		m_target_type = target_type_;
		m_has_targets = has_targets_;

		// register the size of the DataSet
		m_n_examples = n_examples;

		// create the two main arrays of pointers.
		m_examples 	= new Tensor* [m_n_examples];
		if(m_has_targets)
		{
			m_targets = new Tensor* [m_n_examples];
			// set all the target pointers to NULL
			for (int cnt = 0; cnt < m_n_examples; ++ cnt)
			{
				m_targets[cnt]  = NULL;
			}
		}

		// swith over typ of tensor and size of tensor
		switch (example_type_)
		{
		case Tensor::Char:	// Char
			for (int cnt = 0; cnt < m_n_examples; ++ cnt)
			{
				m_examples[cnt] = new CharTensor;
			}
			break;

		case Tensor::Short:	// Short
			for (int cnt = 0; cnt < m_n_examples; ++ cnt)
			{
				m_examples[cnt] = new ShortTensor;
			}
			break;

		case Tensor::Int:	// Int
			for (int cnt = 0; cnt < m_n_examples; ++ cnt)
			{
				m_examples[cnt] = new IntTensor;
			}
			break;

		case Tensor::Long:	// Long
			for (int cnt = 0; cnt < m_n_examples; ++ cnt)
			{
				m_examples[cnt] = new LongTensor;
			}
			break;

		case Tensor::Float:	// Float
			for (int cnt = 0; cnt < m_n_examples; ++ cnt)
			{
				m_examples[cnt] = new FloatTensor;
			}
			break;

		case Tensor::Double:	// Double
			for (int cnt = 0; cnt < m_n_examples; ++ cnt)
			{
				m_examples[cnt] = new DoubleTensor;
			}
			break;

		default:		// ???
			message("MemoryDataSet: sorry target type not supported yet");
			for (int cnt = 0; cnt < m_n_examples; ++ cnt)
			{
				m_examples[cnt] = NULL;
			}
		}

	}


	//////////////////////////////////////////////////////////////////////////////
	// \breif getTarget

	void MemoryDataSet::cleanup()
	{
		if (NULL != m_examples)
		{
			for (int i = 0 ; i < m_n_examples; i++)
				delete m_examples[i];

			delete[] m_examples;
		}

		if (NULL != m_targets)
		{
			delete[] m_targets;
		}
	}
}
