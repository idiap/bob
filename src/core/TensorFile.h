#ifndef	_TORCH5SPRO_TENSOR_FILE_H_
#define _TORCH5SPRO_TENSOR_FILE_H_

#include "File.h"
#include "Tensor.h"

namespace Torch
{
	//////////////////////////////////////////////////////////////////////////////////////
	// Torch::TensorFile:
	//	- object to load and save tensors to some file
	//
	// TODO: doxygen header!
	//////////////////////////////////////////////////////////////////////////////////////

	class TensorFile : public Object
	{
	public:

		struct Header
		{
			Tensor::Type	m_type;
			int		m_n_samples;
			int		m_n_dimensions;
			int		m_size0;
			int		m_size1;
			int		m_size2;
			int		m_size3;
		};

		// Constructor
		TensorFile();

		// Open a data file for reading
		bool 			openRead(const char* file_name);

		// Open a data file for writing
		bool			openWrite(const char* file_name,
						Tensor::Type type, int n_samples,
						int n_dimensions,
						int size0, int size1 = 0, int size2 = 0, int size3 = 0);

		// Open a data file for appending
		bool			openAppend(const char* file_name);

		// Close the file (if opened)
		void			close();

		// Check if the file is opened
		bool			isOpened() const;

		// Load next tensor (returns NULL at the end or some reading error)
		// NB: The tensor should be deallocated outside this class!
		Tensor*			load();

		// Load next tensor to a buffer (returns false at the end or some reading error)
		bool			load(Tensor& tensor);

		// Load a tensor at some index
		Tensor*			load(int index);
		bool			load(Tensor& tensor, int index);

		// Save a tensor to the file (returns false if data mismatch or some writing error)
		bool			save(const Tensor& tensor);

		// Header access
		const Header& 		getHeader() const{ return m_header; }

		///////////////////////////////////////////////////////////

	private:

		///////////////////////////////////////////////////////////
		// Attributes

		File			m_file;			//
		Header			m_header;		// Header information
	};
}

#endif
