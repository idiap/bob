/**
 * @file cxx/core/src/File.cc
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "core/File.h"
#include "core/Tensor.h"

namespace bob
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
	// Read/Writes some tag in the file

	bool File::readTag(const char* tag)
	{
		// Read the tag
		const int tag_size = strlen(tag);
		char* tag_ = new char[tag_size + 1];
		tag_[tag_size] = '\0';
		if (scanf("%s", tag_) != 1)
		{
			bob::message("File: sorry, the tag <%s> cannot be read!", tag);
			delete[] tag_;
			return false;
		}

		// Check the tag
		if (strcmp(tag, tag_) != 0)
		{
			bob::message("File: tag <%s> not found!", tag);
			delete[] tag_;
			return false;
		}
		delete[] tag_;

		// OK
		return true;
	}

	bool File::writeTag(const char* tag)
	{
		if (printf("%s ", tag) != (int)strlen(tag) + 1)
		{
			bob::message("File: sorry, the tag <%s> cannot be written!", tag);
			return false;
		}

		// OK
		return true;
	}

	///////////////////////////////////////////////////////////
	// Reads/Writes some text value in the file

	bool File::readValue(unsigned char* value)
	{
		unsigned int temp;
		if (scanf("%u", &temp) < 1)
		{
			return false;
		}
		*value = (unsigned char)temp;
		return true;
	}
	bool File::readValue(bool* value)
	{
		int temp;
		if (readValue(&temp) == false)
		{
			return false;
		}
		*value = (temp != 0);
		return true;
	}
	bool File::readValue(char* value)
	{
		return scanf("%c", value) > 0;
	}
	bool File::readValue(short* value)
	{
		return scanf("%d", value) > 0;
	}
	bool File::readValue(int* value)
	{
		return scanf("%d", value) > 0;
	}
	bool File::readValue(int64_t* value)
	{
		return scanf("%ld", value) > 0;
	}
	bool File::readValue(float* value)
	{
		return scanf("%f", value) > 0;
	}
	bool File::readValue(double* value)
	{
		return scanf("%lf", value) > 0;
	}

	bool File::writeValue(const unsigned char* value)
	{
		const unsigned int temp = *value;
		return printf("%u ", temp) > 0;
	}
	bool File::writeValue(const bool* value)
	{
		const int temp = (*value == true) ? 1 : 0;
		return printf("%d ", temp) > 0;
	}
	bool File::writeValue(const char* value)
	{
		return printf("%c ", *value) > 0;
	}
	bool File::writeValue(const short* value)
	{
		return printf("%d ", *value) > 0;
	}
	bool File::writeValue(const int* value)
	{
		return printf("%d ", *value) > 0;
	}
	bool File::writeValue(const int64_t* value)
	{
		return printf("%ld ", *value) > 0;
	}
	bool File::writeValue(const float* value)
	{
		return printf("%f ", *value) > 0;
	}
	bool File::writeValue(const double* value)
	{
		return printf("%lf ", *value) > 0;
	}

	///////////////////////////////////////////////////////////
	// Reads and checks the number of elements

	bool File::readNElements(int n)
	{
		int n_ = 0;
		if (readValue(&n_) == false)
		{
			bob::message("File:: cannot read the number of elements!\n");
			return false;
		}
		if (n_ != n)
		{
			bob::message("File: read <%d> elements and was given <%d> elements!\n", n_, n);
			return false;
		}

		return true;
	}

	///////////////////////////////////////////////////////////
	/// Read and check the tag & the number of elements. To be used with #taggedWrite()#.
	///	If the tag and the number of elements read don't correspond an error will occur.

	int File::taggedRead(TensorSize* ptr, const char* tag)
	{
		if (readTag(tag) == false)
		{
			return 0;
		}

		return 	(readValue(&ptr->n_dimensions) == 1 &&
			readValue(&ptr->size[0]) == 1 &&
			readValue(&ptr->size[1]) == 1 &&
			readValue(&ptr->size[2]) == 1 &&
			readValue(&ptr->size[3]) == 1) ? 1 : 0;
	}
	int File::taggedRead(unsigned char* ptr, int n, const char* tag)
	{
		if (readTag(tag) == false || readNElements(n) == false)
		{
			return 0;
		}

		int i = 0;
		for ( ; i < n && readValue(&ptr[i]) == true; i ++)
			;
		return i;
	}
	int File::taggedRead(bool* ptr, int n, const char* tag)
	{
		if (readTag(tag) == false || readNElements(n) == false)
		{
			return 0;
		}

		int i = 0;
		for ( ; i < n && readValue(&ptr[i]) == true; i ++)
			;
		return i;
	}
	int File::taggedRead(char* ptr, int n, const char* tag)
	{
		if (readTag(tag) == false || readNElements(n) == false)
		{
			return 0;
		}

		int i = 0;
		for ( ; i < n && readValue(&ptr[i]) == true; i ++)
			;
		ptr[i] = '\0';
		return i;
	}
	int File::taggedRead(short* ptr, int n, const char* tag)
	{
		if (readTag(tag) == false || readNElements(n) == false)
		{
			return 0;
		}

		int i = 0;
		for ( ; i < n && readValue(&ptr[i]) == true; i ++)
			;
		return i;
	}
	int File::taggedRead(int* ptr, int n, const char* tag)
	{
		if (readTag(tag) == false || readNElements(n) == false)
		{
			return 0;
		}

		int i = 0;
		for ( ; i < n && readValue(&ptr[i]) == true; i ++)
			;
		return i;
	}
	int File::taggedRead(int64_t* ptr, int n, const char* tag)
	{
		if (readTag(tag) == false || readNElements(n) == false)
		{
			return 0;
		}

		int i = 0;
		for ( ; i < n && readValue(&ptr[i]) == true; i ++)
			;
		return i;
	}
	int File::taggedRead(float* ptr, int n, const char* tag)
	{
		if (readTag(tag) == false || readNElements(n) == false)
		{
			return 0;
		}

		int i = 0;
		for ( ; i < n && readValue(&ptr[i]) == true; i ++)
			;
		return i;
	}
	int File::taggedRead(double* ptr, int n, const char* tag)
	{
		if (readTag(tag) == false || readNElements(n) == false)
		{
			return 0;
		}

		int i = 0;
		for ( ; i < n && readValue(&ptr[i]) == true; i ++)
			;
		return i;
	}

	///////////////////////////////////////////////////////////
	/// Write the tag & the number of elements

	int File::taggedWrite(const TensorSize* ptr, const char* tag)
	{
		if (writeTag(tag) == false)
		{
			return 0;
		}

		return (writeValue(&ptr->n_dimensions) == 1 &&
			writeValue(&ptr->size[0]) == 1 &&
			writeValue(&ptr->size[1]) == 1 &&
			writeValue(&ptr->size[2]) == 1 &&
			writeValue(&ptr->size[3]) == 1 &&
			printf("\n") > 0) ? 1 : 0;
	}
	int File::taggedWrite(const unsigned char* ptr, int n, const char* tag)
	{
		if (writeTag(tag) == false || writeValue(&n) == false)
		{
			return 0;
		}

		int i = 0;
		for ( ; i < n && writeValue(&ptr[i]) == true; i ++)
			;
		printf("\n");
		return i;
	}
	int File::taggedWrite(const bool* ptr, int n, const char* tag)
	{
		if (writeTag(tag) == false || writeValue(&n) == false)
		{
			return 0;
		}

		int i = 0;
		for ( ; i < n && writeValue(&ptr[i]) == true; i ++)
			;
		printf("\n");
		return i;
	}
	int File::taggedWrite(const char* ptr, int n, const char* tag)
	{
		if (writeTag(tag) == false || writeValue(&n) == false)
		{
			return 0;
		}

		int i = 0;
		for ( ; i < n && writeValue(&ptr[i]) == true; i ++)
			;
		printf("\n");
		return i;
	}
	int File::taggedWrite(const short* ptr, int n, const char* tag)
	{
		if (writeTag(tag) == false || writeValue(&n) == false)
		{
			return 0;
		}

		int i = 0;
		for ( ; i < n && writeValue(&ptr[i]) == true; i ++)
			;
		printf("\n");
		return i;
	}
	int File::taggedWrite(const int* ptr, int n, const char* tag)
	{
		if (writeTag(tag) == false || writeValue(&n) == false)
		{
			return 0;
		}

		int i = 0;
		for ( ; i < n && writeValue(&ptr[i]) == true; i ++)
			;
		printf("\n");
		return i;
	}
	int File::taggedWrite(const int64_t* ptr, int n, const char* tag)
	{
		if (writeTag(tag) == false || writeValue(&n) == false)
		{
			return 0;
		}

		int i = 0;
		for ( ; i < n && writeValue(&ptr[i]) == true; i ++)
			;
		printf("\n");
		return i;
	}
	int File::taggedWrite(const float* ptr, int n, const char* tag)
	{
		if (writeTag(tag) == false || writeValue(&n) == false)
		{
			return 0;
		}

		int i = 0;
		for ( ; i < n && writeValue(&ptr[i]) == true; i ++)
			;
		printf("\n");
		return i;
	}
	int File::taggedWrite(const double* ptr, int n, const char* tag)
	{
		if (writeTag(tag) == false || writeValue(&n) == false)
		{
			return 0;
		}

		int i = 0;
		for ( ; i < n && writeValue(&ptr[i]) == true; i ++)
			;
		printf("\n");
		return i;
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

