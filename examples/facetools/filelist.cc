#include "FileList.h"

#include "eyecenterGTFile.h"
#include "eyecornerGTFile.h"
#include "bancaGTFile.h"
#include "cootesGTFile.h"
#include "frontalEyeNoseChinGTFile.h"
#include "halfprofileEyeNoseChinGTFile.h"
#include "profileEyeNoseChinGTFile.h"

#include "Image.h"
#include "xtprobeImageFile.h"

#include "CmdLine.h"

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
	int  gt_format;
	bool verbose;
	bool one_gt_object;

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
	cmd.addICmdOption("-gt_format", &gt_format, 1, "gt format (1=eyes center, 2=banca format, 3=eyes corners, 4=eye corners + nose tip + chin, 5=left eye corners + right eye center + nose tip + chin, 6=left eye center + nose tip + chin, 7=Tim Cootes's markup 68 pts)");
	cmd.addBCmdOption("-one_gt_object", &one_gt_object, false, "if true then considers that the gt file contains one object, otherwise assumes that the first line of the file contains the number of objects");
	cmd.addBCmdOption("-verbose", &verbose, false, "verbose");

	// Parse the command line
	if (cmd.read(argc, argv) < 0)
	{
		return 1;
	}

	GTFile *gt_loader = NULL;
	switch(gt_format)
	{
	case 1:
		gt_loader = new eyecenterGTFile();
		break;
	case 2:
		gt_loader = new eyecornerGTFile();
		break;
	case 3:
		gt_loader = new bancaGTFile();
		break;
	case 4:
		gt_loader = new frontalEyeNoseChinGTFile();
		break;
	case 5:
		gt_loader = new halfprofileEyeNoseChinGTFile();
		break;
	case 6:
		gt_loader = new profileEyeNoseChinGTFile();
		break;
	case 7:
		gt_loader = new cootesGTFile();
		break;
	default:
	   	warning("GT format not implemented.");
	   	return 1;
		break;
	}
	gt_loader->setBOption("verbose", verbose);

	if(one_gt_object) message("One object in GT file.");

	FileList *file_list = new FileList(list_filename);

	print("Number of files:%d\n", file_list->n_files);
	for(int i = 0 ; i < file_list->n_files ; i++)
	{
	   	//
		print("%s\n", file_list->file_names[i]);

		char *image_filename = new char [strlen(image_pathname) + strlen(file_list->file_names[i]) + 3 + strlen(image_ext)];
		char *gt_filename = new char [strlen(gt_pathname) + strlen(file_list->file_names[i]) + 3 + strlen(gt_ext)];

		sprintf(image_filename, "%s/%s.%s", image_pathname, file_list->file_names[i], image_ext);
		sprintf(gt_filename, "%s/%s.%s", gt_pathname, file_list->file_names[i], gt_ext);

		//
		print("Image file: %s\n", image_filename);

		Image image(1, 1, 3);
		xtprobeImageFile xtprobe;
		CHECK_FATAL(xtprobe.load(image, image_filename) == true);

		print("   width   = %d\n", image.size(1));
		print("   height  = %d\n", image.size(0));
		print("   nplanes = %d\n", image.size(2));

		//
		print("GT file: %s\n", gt_filename);

		File gt_file;
		gt_file.open(gt_filename, "r");
		if(one_gt_object) gt_loader->load(&gt_file);
		else
		{
		   	int n;
		   	gt_file.scanf("%d", &n);
			print("Number of objects: %d\n", n);
			for(int j = 0 ; j < n ; j++)
				gt_loader->load(&gt_file);
		}
		gt_file.close();

		//
		delete [] gt_filename;
		delete [] image_filename;
	}

	delete file_list;
	delete gt_loader;

        // OK
	return 0;
}

