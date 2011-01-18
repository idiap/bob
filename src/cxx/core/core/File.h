#ifndef	FILE_INC
#define FILE_INC

#include "core/Object.h"

namespace Torch
{
	struct TensorSize;

	/** A file on the disk.
		@author Ronan Collobert (collober@idiap.ch)
	*/
	class File : public Object
	{
	public:

		// Constructor
		File();

		// Destructor
		virtual ~File();

		/// Open "file_name" with the flags #open_flags#
		virtual bool		open(const char* file_name, const char* open_flags);

		/// Use the already given FILE object
		virtual bool		open(FILE* file);

		/// Close the file (if opened)
		void			close();

		/// Check if the file is opened
		bool			isOpened() const;

		/// Read and check the tag & the number of elements. To be used with #taggedWrite()#.
		///	If the tag and the number of elements read don't correspond an error will occur.
		int			taggedRead(TensorSize* ptr, const char* tag);
		int			taggedRead(unsigned char* ptr, int n, const char* tag);
		int			taggedRead(bool* ptr, int n, const char* tag);
		int			taggedRead(char* ptr, int n, const char* tag);
		int			taggedRead(short* ptr, int n, const char* tag);
		int			taggedRead(int* ptr, int n, const char* tag);
		int			taggedRead(int64_t* ptr, int n, const char* tag);
		int			taggedRead(float* ptr, int n, const char* tag);
		int			taggedRead(double* ptr, int n, const char* tag);

		/// Write the tag & the number of elements
		int			taggedWrite(const TensorSize* ptr, const char* tag);
		int			taggedWrite(const unsigned char* ptr, int n, const char* tag);
		int			taggedWrite(const bool* ptr, int n, const char* tag);
		int			taggedWrite(const char* ptr, int n, const char* tag);
		int			taggedWrite(const short* ptr, int n, const char* tag);
		int			taggedWrite(const int* ptr, int n, const char* tag);
		int			taggedWrite(const int64_t* ptr, int n, const char* tag);
		int			taggedWrite(const float* ptr, int n, const char* tag);
		int			taggedWrite(const double* ptr, int n, const char* tag);

		//////////////////////////////////////////////////////////////////////////////////////

		/// Read something.
		int			read(void* ptr, int block_size, int n_blocks);

		/// Write.
		int			write(const void* ptr, int block_size, int n_blocks);

		/// Are we at the end ?
		int			eof();

		/// Flush the file.
		int			flush();

		/// Seek.
		int 			seek(long offset, int whence);

		/// Tell me where am I...
		long			tell();

		/// Rewind.
		void			rewind();

		/// Print some text.
		int 			printf(const char* format, ...);

		/// Scan some text.
		int			scanf(const char* format, void* ptr);

		/// Get one line (read at most #size_# characters).
		char*			gets(char* dest, int size_);

		//////////////////////////////////////////////////////////////////////////////////////

	private:

		// Read/Writes some tag in the file
		bool			readTag(const char* tag);
		bool			writeTag(const char* tag);

		// Reads/Writes some text value in the file
		bool			readValue(unsigned char* value);
		bool			readValue(bool* value);
		bool			readValue(char* value);
		bool			readValue(short* value);
		bool			readValue(int* value);
		bool			readValue(int64_t* value);
		bool			readValue(float* value);
		bool			readValue(double* value);

		bool			writeValue(const unsigned char* value);
		bool			writeValue(const bool* value);
		bool			writeValue(const char* value);
		bool			writeValue(const short* value);
		bool			writeValue(const int* value);
		bool			writeValue(const int64_t* value);
		bool			writeValue(const float* value);
		bool			writeValue(const double* value);

		// Reads and checks the number of elements
		bool			readNElements(int n);

	public:

		///////////////////////////////////////////////////////////
		// Attributes

		FILE*			m_file;
		bool			m_shouldClose;
	};
}

#endif
