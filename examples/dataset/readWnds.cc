#include "torch5spro.h"

using namespace Torch;

//////////////////////////////////////////////////////////////////////////////////////
// Main
//////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
	// Wnd file to read
	char*	wnd_filename;

	// Build the command line object
	CmdLine cmd;
	cmd.setBOption("write log", false);
	cmd.info("Read a .wnd file.");

	cmd.addText("\nArguments:");
	cmd.addSCmdArg(".wnd file", &wnd_filename, ".wnd file");

	// Parse the command line
	if (cmd.read(argc, argv) < 0)
	{
		return 0;
	}

	File fwnd;
	CHECK_FATAL(fwnd.open(wnd_filename, "r") == true);

	// Read all the subwindows
	int n_samples = 0;
	CHECK_FATAL(fwnd.read(&n_samples, sizeof(int), 1) == 1);
	for (int i = 0; i < n_samples; i ++)
	{
		short x, y, w, h;

		CHECK_FATAL(fwnd.read(&x, sizeof(short), 1) == 1);
		CHECK_FATAL(fwnd.read(&y, sizeof(short), 1) == 1);
		CHECK_FATAL(fwnd.read(&w, sizeof(short), 1) == 1);
		CHECK_FATAL(fwnd.read(&h, sizeof(short), 1) == 1);

		print("\t[%d/%d]: (%d, %d) - %dx%d\n", i + 1, n_samples, x, y, w, h);
	}

	fwnd.close();

   	return 0;
}

