#include "DiskDataSet.h"
#include "Tensor.h"

namespace Torch {

DiskDataSet::DiskDataSet(Tensor::Type example_type_)
	: 	DataSet(example_type_),
		m_buffer(0), m_targets(0),
		m_files(0),
		m_n_files(0)
{
}

DiskDataSet::~DiskDataSet()
{
	cleanup();
}

void DiskDataSet::cleanup()
{
	delete m_buffer;
	delete[] m_targets;

	for (int i = 0; i < m_n_files; i ++)
	{
		delete m_files[i];
	}
	delete[] m_files;
}

Tensor* DiskDataSet::getExample(long index)
{
	if(!isIndex(index, m_n_examples))
		error("DiskDataSet::getExample - target (%d) out-of-range [0-%d].", index, m_n_examples - 1);

	// Search the file where this example is written in
	int ifile = 0;
	while (	ifile < m_n_files &&
		index >= m_files[ifile]->getHeader().m_n_samples)
	{
		index -= m_files[ifile]->getHeader().m_n_samples;
		ifile ++;
	}

	// Store the tensor data into the buffer, from the selected file
	if (m_buffer == 0)
	{
		m_buffer = m_files[ifile]->load(index);
		if (m_buffer == NULL)
		{
			message("DiskDataSet::getExample - error loading from file!\n");
			return NULL;
		}
	}
	else
	{
		if (m_files[ifile]->load(*m_buffer, index) == false)
		{
			message("DiskDataSet::getExample - error loading from file!\n");
			return NULL;
		}
	}

	// OK
	return m_buffer;
}

Tensor& Torch::DiskDataSet::operator()(long index)
{
	return *getExample(index);
}

Tensor* Torch::DiskDataSet::getTarget(long index)
{
	if(m_targets == NULL)
		error("DiskDataSet(): no examples in memory.");

	if(!isIndex(index, m_n_examples))
		error("DiskDataSet(): target (%d) out-of-range [0-%d].", index, m_n_examples - 1);

	return m_targets[index];
}

void Torch::DiskDataSet::setTarget(long index, Tensor* target)
{
	if(m_targets == NULL)
		error("DiskDataSet(): no examples in memory.");

	if(!isIndex(index, m_n_examples))
		error("DiskDataSet(): target (%d) out-of-range [0-%d].", index, m_n_examples - 1);

	m_targets[index] = target;
}

bool DiskDataSet::load(const char* filename)
{
	// Load the file, check its header
	TensorFile* tf = new TensorFile;
	if (	tf->openRead(filename) == false ||
		tf->getHeader().m_type != m_example_type)
	{
		delete tf;
		return false;
	}

	// Add the opened file
	TensorFile** new_files = new TensorFile*[m_n_files + 1];
	for (int i = 0; i < m_n_files; i ++)
	{
		new_files[i] = m_files[i];
	}
	new_files[m_n_files ++] = tf;
	delete[] m_files;
	m_files = new_files;

	// Make room for new targes
	Tensor** new_targets = new Tensor*[m_n_examples + tf->getHeader().m_n_samples];
	for (int i = 0; i < m_n_examples; i ++)
	{
		new_targets[i] = m_targets[i];
	}
	for (int i = 0; i < tf->getHeader().m_n_samples; i ++)
	{
		new_targets[i + m_n_examples] = NULL;
	}
	delete[] m_targets;
	m_targets = new_targets;

	// OK
	m_n_examples += tf->getHeader().m_n_samples;
	return true;
}

} // namespace torch
