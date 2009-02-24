#include "FileList.h"
#include "File.h"

namespace Torch {

FileList::FileList(const char *name_)
{
	n_files = 0;
	file_names = NULL;
	
	if (name_ == NULL)
		Torch::error("FileList: no file provided");
	else
	{
		/// Read the contents of the file...
		File file;
		if (file.open(name_, "r") == false)
		{
			error("FileList: cannot open file <%s>", name_);
		}

		char str[1204];
		file.read(str, 1, 1024);
		str[1023] = '\0';
		file.rewind();

		char* endp_;
		strtol(str, &endp_, 10);
		if ( (*endp_ != '\0') && (*endp_ != '\n') )
		{
			do
			{
				file.gets(str, 1024);
				n_files++;
			} while (!file.eof());
			n_files--;
			file.rewind();
		}
		else file.scanf("%d", &n_files);

		//Torch::message("FileList: %d files detected", n_files);

		if(n_files > 0)
		{
			// read the file
			file_names = new char*[n_files];
			for (int i = 0; i < n_files; i++)
			{
				file.scanf("%s", str);
				file_names[i] = new char[strlen(str)+1];
				strcpy(file_names[i], str);
			}
		}
	}
}

FileList::~FileList()
{
   	if(file_names != NULL)
	{
		for (int i = 0; i < n_files; i ++)
			delete[] file_names[i];
		delete[] file_names;
	}
}

}
