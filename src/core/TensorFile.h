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

	class TensorFile
	{
	public:

		// Header information
		struct Header
		{
			// Get the index of some tensor in the file
			int		getTensorIndex(int tensor_index);

			// Update internal information
			void		update();

			// Attributes
			Tensor::Type	m_type;
			int		m_n_samples;
			int		m_n_dimensions;
			int		m_size[4];
			int		m_tensorSize;
		};

		// Constructor
		TensorFile();

		// Destructor
		~TensorFile();

		// Open a data file for reading / writing and appending
		bool 			openRead(const char* file_name);
		bool			openWrite(	const char* file_name,
							Tensor::Type type,
							int n_dimensions,
							int size0, int size1 = 0, int size2 = 0, int size3 = 0);
		bool			openAppend(const char* file_name);

		// Close the file (if opened)
		void			close();

		// Check if the file is opened
		bool			isOpened() const;

		// Load next tensor (returns NULL at the end or some reading error)
		// NB: The tensor should be deallocated outside this class!
		Tensor*			load();
		Tensor*			load(int index);			// Load some tensor index

		// Load next tensor to a buffer (returns false at the end or some reading error)
		bool			load(Tensor& tensor);
		bool			load(Tensor& tensor, int index);	// Load some tensor index

		// Save a tensor to the file (returns false if data mismatch or some writing error)
		bool			save(const Tensor& tensor);

		// Header access
		const Header& 		getHeader() const{ return m_header; }

		///////////////////////////////////////////////////////////

	private:

		///////////////////////////////////////////////////////////
		// Header processing functions

		bool			read_header();
		bool			write_header();
		bool			header_ok();

		///////////////////////////////////////////////////////////
		// Attributes

		enum Mode
		{
			Read,
			Write
		};

		File			m_file;			// File to operate on
		Header			m_header;		// Header information
		Mode 			m_mode;			// Opening mode
	};
}

#endif
