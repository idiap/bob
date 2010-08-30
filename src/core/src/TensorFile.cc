#include "core/TensorFile.h"

namespace Torch
{

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Get the index of some tensor in the file

int TensorFile::Header::getTensorIndex(int tensor_index)
{
	static const int header_size = 7 * sizeof(int);

	return header_size + tensor_index * m_tensorSize;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Update internal information

void TensorFile::Header::update()
{
	int base_size = 0;
	switch (m_type)
	{
	case Tensor::Char:
		base_size = sizeof(char);
		break;

	case Tensor::Short:
		base_size = sizeof(short);
		break;

	case Tensor::Int:
		base_size = sizeof(int);
		break;

	case Tensor::Long:
		base_size = sizeof(long);
		break;

	case Tensor::Float:
		base_size = sizeof(float);
		break;

	case Tensor::Double:
		base_size = sizeof(double);
		break;

  default:
    break;
	}

	int tsize = 1;
	for (int i = 0; i < m_n_dimensions; i ++)
	{
		tsize *= m_size[i];
	}

	m_tensorSize = tsize * base_size;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Constructor

TensorFile::TensorFile()
	:	m_header(),
		m_mode(Read)
{
}

TensorFile::~TensorFile()
{
	close();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Open a data file for reading

bool TensorFile::openRead(const char* file_name)
{
	close();

	// Open the file and read its header
	if (	m_file.open(file_name, "r") == false ||
		read_header() == false)
	{
		m_file.close();
		return false;
	}

	// OK
	m_mode = Read;
	return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Open a data file for writing

bool TensorFile::openWrite(	const char* file_name, Tensor::Type type, int n_dimensions,
				int size0, int size1, int size2, int size3)
{
	close();

	// Save the header information: type, number of samples, dimensions, sizes
	m_header.m_type = type;
	m_header.m_n_samples = 0;
        m_header.m_n_dimensions = n_dimensions;
        m_header.m_size[0] = size0;
        m_header.m_size[1] = size1;
        m_header.m_size[2] = size2;
        m_header.m_size[3] = size3;

	// Open the file and write its header
	if (	m_file.open(file_name, "w") == false ||
		write_header() == false)
	{
		m_file.close();
		return false;
	}

	// OK
	m_mode = Write;
	return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Open a data file for appending

bool TensorFile::openAppend(const char* file_name)
{
	close();

	// Open the file and read its header
	if (	m_file.open(file_name, "r+") == false ||
		read_header() == false)
	{
		m_file.close();
		return false;
	}

	// move to the end of file so we can append
	m_file.seek(0, SEEK_END);

	// OK
	m_mode = Write;
	return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Close the file (if opened)

void TensorFile::close()
{
	if (isOpened() == true)
	{
		// write header since it might be updated
		if (m_mode == Write)
		{
			write_header();
		}

		// close the file
		m_file.close();
	}

	m_mode = Read;
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
	// Make local copies to make it easier to follow
	const int p1 = m_header.m_size[0];
	const int p2 = m_header.m_size[1];
	const int p3 = m_header.m_size[2];
	const int p4 = m_header.m_size[3];

	// Create new tensor of type specified in header
	Tensor* tns = 0;

	// Switch over type of tensor and size of tensor
	switch (m_header.m_type)
	{
	case Tensor::Char:
		switch (m_header.m_n_dimensions)
		{
		case 1:
			tns = new CharTensor(p1);
			break;
		case 2:
			tns = new CharTensor(p1, p2);
			break;
		case 3:
			tns = new CharTensor(p1, p2, p3);
			break;
		case 4:
			tns = new CharTensor(p1, p2, p3, p4);
			break;
		}
		break;

	case Tensor::Short:
		switch (m_header.m_n_dimensions)
		{
		case 1:
			tns = new ShortTensor(p1);
			break;
		case 2:
			tns = new ShortTensor(p1, p2);
			break;
		case 3:
			tns = new ShortTensor(p1, p2, p3);
			break;
		case 4:
			tns = new ShortTensor(p1, p2, p3, p4);
			break;
		}
		break;

	case Tensor::Int:
		switch (m_header.m_n_dimensions)
		{
		case 1:
			tns = new IntTensor(p1);
			break;
		case 2:
			tns = new IntTensor(p1, p2);
			break;
		case 3:
			tns = new IntTensor(p1, p2, p3);
			break;
		case 4:
			tns = new IntTensor(p1, p2, p3, p4);
			break;
		}
		break;

	case Tensor::Long:
		switch (m_header.m_n_dimensions)
		{
		case 1:
			tns = new LongTensor(p1);
			break;
		case 2:
			tns = new LongTensor(p1, p2);
			break;
		case 3:
			tns = new LongTensor(p1, p2, p3);
			break;
		case 4:
			tns = new LongTensor(p1, p2, p3, p4);
			break;
		}
		break;

	case Tensor::Float:
		switch(m_header.m_n_dimensions)
		{
		case 1:
			tns = new FloatTensor(p1);
			break;
		case 2:
			tns = new FloatTensor(p1, p2);
			break;
		case 3:
			tns = new FloatTensor(p1, p2, p3);
			break;
		case 4:
			tns = new FloatTensor(p1, p2, p3, p4);
			break;
		}
		break;

	case Tensor::Double:
		switch(m_header.m_n_dimensions)
		{
		case 1:
			tns = new DoubleTensor(p1);
			break;
		case 2:
			tns = new DoubleTensor(p1, p2);
			break;
		case 3:
			tns = new DoubleTensor(p1, p2, p3);
			break;
		case 4:
			tns = new DoubleTensor(p1, p2, p3, p4);
			break;
		}
		break;

	default:
		// error
		tns = 0;
	}

	// Make sure that we have successfully allocated a tensor
	if (0 == tns)
	{
		return 0;
	}

	// Try to load in information into tensor
	if (load(*tns) == false)
	{
		delete tns;
		return 0;
	}

	// OK
	return tns;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Load a tensor with a given index

Tensor* TensorFile::load(int index)
{
	// Check the index
	if (	isOpened() == false ||
		isIndex(index, m_header.m_n_samples) == false)
	{
		return false;
	}

	// Position at the requested index and return the tensor
	m_file.seek(m_header.getTensorIndex(index), SEEK_SET);
	return load();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Load next tensor to a buffer (returns false at the end or some reading error)

bool TensorFile::load(Tensor& tensor)
{
	// Make sure file is open
	if (!isOpened())
	{
		return false;
	}

	// Make sure that the input tensor has the same properties as the header
	if (	m_header.m_type != tensor.getDatatype() ||
		m_header.m_n_dimensions != tensor.nDimension())
	{
		return false;
	}
	for (int i = 0; i < m_header.m_n_dimensions; i ++)
		if (m_header.m_size[i] != tensor.size(i))
		{
			return false;
		}

	// Read up tensor
	if (m_file.read(tensor.dataW(), tensor.typeSize(), tensor.sizeAll()) != tensor.sizeAll())
	{
		return false;
	}

	// OK
	return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Load a tensor with a given index

bool TensorFile::load(Tensor& tensor, int index)
{
	// Check the index
	if (	isOpened() == false ||
		isIndex(index, m_header.m_n_samples) == false)
	{
		return false;
	}

	// Position at the requested index and return the tensor
	m_file.seek(m_header.getTensorIndex(index), SEEK_SET);
	return load(tensor);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Save a tensor to the file (returns false if data mismatch or some writing error)

bool TensorFile::save(const Tensor& tensor)
{
	// Make sure file is open
	if (!isOpened())
	{
		return false;
	}

	// Make sure that the tensor have the same properties as the header
	if ( 	tensor.getDatatype() 	!= m_header.m_type 	||
		tensor.nDimension() 	!= m_header.m_n_dimensions )
	{
	   	warning("TensorFile::save() incorrect data type or number of dimensions.");

		//print(" > %d == %d ?\n", tensor.getDatatype(), m_header.m_type);
		//print(" > %d == %D ?\n", tensor.nDimension(), m_header.m_n_dimensions);

		return false;
	}

	// Make sure that the sizes in each dimension is correct
	for (int cnt = 0; cnt < m_header.m_n_dimensions; cnt ++)
	{
		if ( tensor.size(cnt) != m_header.m_size[cnt] )
		{
	   		warning("TensorFile::save() incorrect sizes.");

			//print(" > %d == %d ?\n", tensor.size(cnt), m_header.m_size[cnt]);

			return false;
		}
	}

	// Write down tensor
	if (m_file.write(tensor.dataR(), tensor.typeSize(), tensor.sizeAll()) != tensor.sizeAll())
	{
	   	warning("TensorFile::save() incorrect write.");
		return false;
	}

	// Update header - new tensor was written
	m_header.m_n_samples ++;

	// OK
	return true;
}

///////////////////////////////////////////////////////////
// Header processing functions

bool TensorFile::read_header()
{
	// Read the header
	m_file.rewind();
	int type = 0;
	if (	m_file.read(&type,  sizeof(int), 1) != 1 ||
		m_file.read(&m_header.m_n_samples, sizeof(int), 1) != 1 ||
		m_file.read(&m_header.m_n_dimensions, sizeof(int), 1) != 1 ||
		m_file.read(&m_header.m_size[0], sizeof(int), 1) != 1 ||
		m_file.read(&m_header.m_size[1], sizeof(int), 1) != 1 ||
		m_file.read(&m_header.m_size[2], sizeof(int), 1) != 1 ||
		m_file.read(&m_header.m_size[3], sizeof(int), 1) != 1)
	{
		m_file.close();
		return false;
	}
	m_header.m_type = (Tensor::Type)type;

	// make sure that the header is ok
	return header_ok();
}

bool TensorFile::write_header()
{
	// Write the header
	m_file.rewind();
	if (	m_file.write(&m_header.m_type,  sizeof(int), 1) != 1 ||
		m_file.write(&m_header.m_n_samples, sizeof(int), 1) != 1 ||
		m_file.write(&m_header.m_n_dimensions, sizeof(int), 1) != 1 ||
		m_file.write(&m_header.m_size[0], sizeof(int), 1) != 1 ||
		m_file.write(&m_header.m_size[1], sizeof(int), 1) != 1 ||
		m_file.write(&m_header.m_size[2], sizeof(int), 1) != 1 ||
		m_file.write(&m_header.m_size[3], sizeof(int), 1) != 1)
	{
		m_file.close();
		return false;
	}

	return true;
}

bool TensorFile::header_ok()
{
	// Check the type
	switch (m_header.m_type)
	{
		// supported tensor types
	case Tensor::Char:
	case Tensor::Short:
	case Tensor::Int:
	case Tensor::Long:
	case Tensor::Float:
	case Tensor::Double:
		break;

		// error
	default:
		return false;
	}

	// Check the number of samples and dimensions
	if (	m_header.m_n_samples < 0 ||
		m_header.m_n_dimensions < 1 ||
		m_header.m_n_dimensions > 4)
	{
		return false;
	}

	// OK
	m_header.update();
	return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////

}
