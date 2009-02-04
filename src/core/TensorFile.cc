#include "TensorFile.h"

namespace Torch
{
/*
//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Constructor

TensorFile::TensorFile()
	:	m_file(),
		m_header_type(Tensor::Short),
		m_header_n_samples(0),
		m_header_n_dimensions(0),
		m_header_size0(0),
		m_header_size1(0),
		m_header_size2(0),
		m_header_size3(0)
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Open a data file for reading

bool TensorFile::openRead(const char* file_name)
{
	if (m_file.open(file_name, "r") == false)
	{
		print("Tensor::openRead - cannot open file [%s]!\n", file_name);
		return false;
	}

	// Read the header - tensor type
	int type = 0;
	if (m_file.read(&type, sizeof(int), 1) != 1)
	{
		print("Tensor::openRead - cannot read the type from file [%s]!\n", file_name);
		m_file.close();
		return false;
	}
	switch (type)
	{
	case Tensor::Char:
	case Tensor::Short:
	case Tensor::Int:
	case Tensor::Long:
	case Tensor::Float:
	case Tensor::Double:
		break;

	default:
		print("Tensor::openRead - invalid type from file [%s]!\n", file_name);
		m_file.close();
		return false;
	}
	m_header_type = (Tensor::Type)type;

	// Read the header - number of samples and number of dimensions
	if (	m_file.read(&m_header_n_samples, sizeof(int), 1) != 1 ||
		m_header_n_samples < 1)
	{
		print("Tensor::openRead - invalid number of samples from file [%s]!\n", file_name);
		m_file.close();
		return false;
	}
	if (	m_file.read(&m_header_n_dimensions, sizeof(int), 1) != 1 ||
		m_header_n_dimensions != getInRange(m_header_n_dimensions, 1, 4))
	{
		print("Tensor::openRead - invalid number of dimensions from file [%s]!\n", file_name);
		m_file.close();
		return false;
	}

	// Read the header - size for each dimension
	int* sizes[4] = { &m_header_size0, &m_header_size1, &m_header_size2, &m_header_size3 };
	for (int i = 0; i < m_header_n_dimensions; i ++)
	{
		if (	m_file.read(sizes[i], sizeof(int), 1) != 1 ||
			*sizes[i] < 1)
		{
			print("Tensor::openRead - invalid size [%d] from file [%s]!\n", i + 1, file_name);
			m_file.close();
			return false;
		}
	}

	// OK
	return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Open a data file for writing

bool TensorFile::openWrite(const char* file_name, Tensor::Type type, int n_samples,
				int n_dimensions, int size0, int size1, int size2, int size3)
{
	if (m_file.open(file_name, "w") == false)
	{
		print("Tensor::openWrite - cannot open file [%s]!\n", file_name);
		return false;
	}

	// Write the header
	if (	m_file.write(&type, sizeof(int), 1) != 1 ||
		m_file.write(&n_samples, sizeof(int), 1) != 1 ||
		m_file.write(&n_dimensions, sizeof(int), 1) != 1)
	{
		print("Tensor::openWrite - cannot read the type from file [%s]!\n", file_name);
		m_file.close();
		return false;
	}
	// TODO: write the sizes!

	// Copy the header information: type, number of samples, dimensions, sizes
	m_header_type = (Tensor::Type)type;
	// TODO

	// OK
	return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Open a data file for appending

bool TensorFile::openAppend(const char* file_name)
{
	// TODO: the same as <openWrite>, but open with <a> flag and go to the end of the file!

	// OK
	return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Close the file (if opened)

void TensorFile::close()
{
	m_file.close();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Check if the file is opened

bool TensorFile::isOpened() const
{
	return m_file.isOpened();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Load next tensor (returns NULL at the end or some reading error)
// NB: The tensor should be deallocated outside this class!

Tensor* TensorFile::load()
{
	// TODO: Allocate a new tensor, and called the <load> below with that pointer and return the pointer

	return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Load next tensor to a buffer (returns false at the end or some reading error)

bool TensorFile::load(Tensor& tensor)
{
	// TODO: check if it's open and the type is ok!

	// OK
	return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Save a tensor to the file (returns false if data mismatch or some writing error)

bool TensorFile::save(const Tensor& tensor)
{
	// TODO

	// OK
	return true;
}
*/
//////////////////////////////////////////////////////////////////////////////////////////////////////////

}

