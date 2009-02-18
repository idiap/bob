#include "CmdLine.h"
#include "FileListCmdOption.h"

using namespace Torch;


int main(int argc, char* argv[])
{
        ///////////////////////////////////////////////////////////////////
        // Parse the command line
        ///////////////////////////////////////////////////////////////////

	// Set options
	FileListCmdOption file_list("file name", "the list files or one data file");
	file_list.isArgument(true);

	bool verbose;

	// Build the command line object
	CmdLine cmd;
	cmd.setBOption("write log", false);

	cmd.info("File List testing program");

	cmd.addText("\nArguments:");
	cmd.addCmdOption(&file_list);

	cmd.addText("\nOptions:");
	cmd.addBCmdOption("-verbose", &verbose, false, "print Tensor values");

	// Parse the command line
	if (cmd.read(argc, argv) < 0)
	{
		return 0;
	}

        // OK
	return 0;
}

