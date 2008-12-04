#ifndef	FILE_INC
#define FILE_INC

#include "Object.h"

namespace Torch
{
	/** A file on the disk.
		@author Ronan Collobert (collober@idiap.ch)
	*/
	class File : public Object
	{
	public:

		// Constructor
		File();

		// Destructor
		~File();

		/// Open "file_name" with the flags #open_flags#
		virtual bool		open(const char* file_name, const char* open_flags);

		/// Use the already given FILE object
		virtual bool		open(FILE* file);

		/// Close the file (if opened)
		void			close();

		/// Check if the file is opened
		bool			isOpened() const;

		/// Read and check the tag/the size. To be used with #taggedWrite()#.
		///	If the tag and the size readed doesn't correspond to the given
		///	tag and size, an error will occur.
		int			taggedRead(void* ptr, int block_size, int n_blocks, const char* tag);

		/// Write and write the tag/the size.
		int			taggedWrite(const void* ptr, int block_size, int n_blocks, const char* tag);

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

	protected:

		///////////////////////////////////////////////////////////
		// Attributes

		FILE*			m_file;
		bool			m_shouldClose;
	};
}

#endif
