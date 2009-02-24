#include "CmdLine.h"
#include "FileList.h"

using namespace Torch;


int main(int argc, char* argv[])
{
        ///////////////////////////////////////////////////////////////////
        // Parse the command line
        ///////////////////////////////////////////////////////////////////

	char *list_filename;
	char *image_pathname;
	char *gt_pathname;
	char *image_ext;
	char *gt_ext;
	bool verbose;

	// Build the command line object
	CmdLine cmd;
	cmd.setBOption("write log", false);

	cmd.info("File List testing program");

	cmd.addText("\nArguments:");
	cmd.addSCmdArg("list_filename", &list_filename, "list of files");
	cmd.addSCmdArg("image_pathname", &image_pathname, "path to image files");
	cmd.addSCmdArg("gt_pathname", &gt_pathname, "path to gt files");

	cmd.addText("\nOptions:");
	cmd.addSCmdOption("-image_ext", &image_ext, "pgm", "image file extension");
	cmd.addSCmdOption("-gt_ext", &gt_ext, "pos", "gt file extension");
	cmd.addBCmdOption("-verbose", &verbose, false, "verbose");

	// Parse the command line
	if (cmd.read(argc, argv) < 0)
	{
		return 0;
	}

	FileList *file_list = new FileList(list_filename);
	
	print("Number of files:%d\n", file_list->n_files);
	for(int i = 0 ; i < file_list->n_files ; i++)
	{
	   	//
		print("%s\n", file_list->file_names[i]);

		char *image_filename = new char [strlen(image_pathname) + strlen(file_list->file_names[i])];
		char *gt_filename = new char [strlen(gt_pathname) + strlen(file_list->file_names[i])];

		sprintf(image_filename, "%s/%s.%s", image_pathname, file_list->file_names[i], image_ext);
		sprintf(gt_filename, "%s/%s.%s", gt_pathname, file_list->file_names[i], gt_ext);

		print("> %s\n", image_filename);
		print("> %s\n", gt_filename);

		delete [] gt_filename;
		delete [] image_filename;
	}

	delete file_list;

        // OK
	return 0;
}

