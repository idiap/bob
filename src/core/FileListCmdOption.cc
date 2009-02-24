#include "FileListCmdOption.h"
#include "File.h"

namespace Torch {

FileListCmdOption::FileListCmdOption(const char *name_, const char *help_, bool save_)
		: CmdOption(name_, "<[-one_file] file_name>", help_, save_)
{
	n_files = 0;
	file_names = NULL;
}

void FileListCmdOption::read(int *argc_, char ***argv_)
{
	char **argv = *argv_;

	if (*argc_ == 0)
		error("FileListCmdOption: cannot correctly set <%s>", name);

	// Special case...
	if (!strcmp("-one_file", argv[0]))
	{
		(*argc_)--;
		(*argv_)++;

		argv = *argv_;
		if (*argc_ == 0)
			error("FileListCmdOption: cannot correctly set <%s>", name);

		n_files = 1;
		file_names = new char*[1];
		file_names[0] = new char[strlen(argv[0])+1];
		strcpy(file_names[0], argv[0]);

		(*argc_)--;
		(*argv_)++;
		return;
	}

	/// Read the contents of the file...
	File file;
	if (file.open(argv[0], "r") == false)
	{
		error("FileListCmdOption: cannot open file <%s>", argv[0]);
	}

	char melanie[1204];
	file.read(melanie, 1, 1024);
	melanie[1023] = '\0';
	file.rewind();

	char* endp_;
	strtol(melanie, &endp_, 10);
	if ( (*endp_ != '\0') && (*endp_ != '\n') )
	{
		do
		{
			file.gets(melanie, 1024);
			n_files++;
		} while (!file.eof());
		n_files--;
		file.rewind();
	}
	else
		file.scanf("%d", &n_files);

	//message("FileListCmdOption: %d files detected", n_files);

	file_names = new char*[n_files];
	for (int i = 0; i < n_files; i++)
	{
		file.scanf("%s", melanie);
		file_names[i] = new char[strlen(melanie)+1];
		strcpy(file_names[i], melanie);
	}

	////////////////////////////////////

	(*argc_)--;
	(*argv_)++;
}

bool FileListCmdOption::loadFile(File& file)
{
	file.taggedRead(&n_files, sizeof(int), 1, "NFILES");
	file_names = new char*[n_files];
	for (int i = 0; i < n_files; i++)
	{
		int melanie;
		file.taggedRead(&melanie, sizeof(int), 1, "SIZE");
		file_names[i] = new char[melanie];
		file.taggedRead(file_names[i], 1, melanie, "FILE");
	}

	return true;
}

bool FileListCmdOption::saveFile(File& file) const
{
	file.taggedWrite(&n_files, sizeof(int), 1, "NFILES");
	for (int i = 0; i < n_files; i++)
	{
		int melanie = strlen(file_names[i])+1;
		file.taggedWrite(&melanie, sizeof(int), 1, "SIZE");
		file.taggedWrite(file_names[i], 1, melanie, "FILE");
	}

	return true;
}

FileListCmdOption::~FileListCmdOption()
{
   	if(file_names != NULL)
	{
		for (int i = 0; i < n_files; i ++)
			delete[] file_names[i];
		delete[] file_names;
	}
}

}
