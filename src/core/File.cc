#include "File.h"

namespace Torch
{
	///////////////////////////////////////////////////////////
	// Constructor

	File::File()
		:	m_file(0)
	{
	}

	///////////////////////////////////////////////////////////
	// Destructor

	File::~File()
	{
		close();
	}

	///////////////////////////////////////////////////////////
	// Open "file_name" with the flags #open_flags#

	bool File::open(const char* file_name, const char* open_flags)
	{
		close();

		m_file = fopen(file_name, open_flags);
		if (m_file != 0)
		{
		        fseek(m_file, 0, SEEK_SET);
		}
		m_shouldClose = true;
		return isOpened();
	}

	///////////////////////////////////////////////////////////
	// Use the already given FILE object

	bool File::open(FILE* file)
	{
                if (file == 0)
                {
                        return false;
                }

                close();

		// If it's an external FILE don't close it, it may be managed externally!
                m_file = file;
                m_shouldClose = false;
                return true;
	}

	///////////////////////////////////////////////////////////
	// Close the file (if opened)

	void File::close()
	{
		if (m_file != 0 && m_shouldClose == true)
		{
			fclose(m_file);
			m_file = 0;
		}
	}

	///////////////////////////////////////////////////////////
	// Check if the file is opened

	bool File::isOpened() const
	{
		return m_file != 0;
	}

	///////////////////////////////////////////////////////////
	// Read and check the tag/the size. To be used with #taggedWrite()#.
	//	If the tag and the size readed doesn't correspond to the given
	//	tag and size, an error will occur.

	int File::taggedRead(void* ptr, int block_size, int n_blocks, const char* tag)
	{
		// TODO: check that the <read>s actually read all the bytes correctly

		int tag_size = 0;
		read(&tag_size, sizeof(int), 1);
		if (tag_size != (int)strlen(tag))
		{
			Torch::message("File: sorry, the tag <%s> cannot be read!", tag);
			return 0;
		}

		char* tag_ = new char[tag_size + 1];
		tag_[tag_size] = '\0';
		read(tag_, 1, tag_size);

		if (strcmp(tag, tag_) != 0)
		{
			Torch::message("XFile: tag <%s> not found!", tag);
			delete tag_;
			return 0;
		}
		delete[] tag_;

		int block_size_;
		int n_blocks_;
		read(&block_size_, sizeof(int), 1);
		read(&n_blocks_, sizeof(int), 1);

		if( (block_size_ != block_size) || (n_blocks_ != n_blocks) )
		{
			Torch::message("XFile: tag <%s> has a corrupted size!", tag);
			return 0;
		}

		return read(ptr, block_size, n_blocks);
	}

	///////////////////////////////////////////////////////////
	// Write and write the tag/the size.

	int File::taggedWrite(const void* ptr, int block_size, int n_blocks, const char* tag)
	{
		// TODO: check that the first 4 <write>s actually write all the bytes correctly

		int tag_size = strlen(tag);
		write(&tag_size, sizeof(int), 1);
		write((char *)tag, 1, tag_size);
		write(&block_size, sizeof(int), 1);
		write(&n_blocks, sizeof(int), 1);
		return write(ptr, block_size, n_blocks);
	}

	///////////////////////////////////////////////////////////
	// Wrappers over C file descriptor

	int File::read(void* ptr, int block_size, int n_blocks)
	{

		return fread(ptr, block_size, n_blocks, m_file);
	}

	int File::write(const void* ptr, int block_size, int n_blocks)
	{
		return fwrite(ptr, block_size, n_blocks, m_file);
	}

	int File::eof()
	{
		return feof(m_file);
	}

	int File::flush()
	{
		return fflush(m_file);
	}

	int File::seek(long offset, int whence)
	{
		return fseek(m_file, offset, whence);
	}

	long File::tell()
	{
		return ftell(m_file);
	}

	void File::rewind()
	{
		::rewind(m_file);
	}

	int File::printf(const char* format, ...)
	{
		va_list args;
		va_start(args, format);
		int res = vfprintf(m_file, format, args);
		va_end(args);
		return res;
	}

	int File::scanf(const char* format, void* ptr)
	{
		return fscanf(m_file, format, ptr);
	}

	char* File::gets(char* dest, int size_)
	{
		return fgets(dest, size_, m_file);
	}

	///////////////////////////////////////////////////////////
}

