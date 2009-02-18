#include "CmdLine.h"
//#include "FileListCmdOption.h"
#include "FileList.h"

using namespace Torch;


int main(int argc, char* argv[])
{
        ///////////////////////////////////////////////////////////////////
        // Parse the command line
        ///////////////////////////////////////////////////////////////////

	// Set options
	//FileListCmdOption *file_list = new FileListCmdOption("file name", "the list files or one data file");
	//file_list->isArgument(true);
	char *list_filename;
	bool verbose;

	// Build the command line object
	CmdLine cmd;
	cmd.setBOption("write log", false);

	cmd.info("File List testing program");

	cmd.addText("\nArguments:");
	//cmd.addCmdOption(file_list);
	cmd.addSCmdArg("list_filename", &list_filename, "list of files");

	cmd.addText("\nOptions:");
	cmd.addBCmdOption("-verbose", &verbose, false, "print Tensor values");

	// Parse the command line
	if (cmd.read(argc, argv) < 0)
	{
		return 0;
	}

	FileList *file_list = new FileList(list_filename);
	
	print("Number of files:%d\n", file_list->n_files);
	for(int i = 0 ; i < file_list->n_files ; i++)
	{
		print("> %s\n", file_list->file_names[i]);
	}

	delete file_list;

        // OK
	return 0;
}

