#include "CmdLine.h"
#include "File.h"
#include <time.h>

namespace Torch {

//////////////////////////////////////////////////////////////////////////////////////////
// Constructor

CmdLine::CmdLine()
{
	n_master_switches = 1; // the default!
	n_cmd_options = new int[1];
	cmd_options = new CmdOption**[1];
	n_cmd_options[0] = 0;
	cmd_options[0] = NULL;

	text_info = NULL;
	working_directory = new char[2];
	strcpy(working_directory, ".");

	associated_files = NULL;
	n_associated_files = 0;

	master_switch = -1;

	program_name = new char[1];
	*program_name = '\0';

	addBOption("write log", false, "Should I output the cmd.log file ?");
}

//////////////////////////////////////////////////////////////////////////////////////////
// Destructor

CmdLine::~CmdLine()
{
	cleanup();
}

//////////////////////////////////////////////////////////////////////////////////////////
// Deallocate memory

void CmdLine::cleanup()
{
	// Delete command line options
        for (int i = 0; i < n_master_switches; i ++)
        {
		const int size = n_cmd_options[i];
		for (int j = 0; j < size; j ++)
        	{
        		delete cmd_options[i][j];
        	}
        	delete[] cmd_options[i];
        }
        delete[] n_cmd_options;
        delete[] cmd_options;

	// Delete strings
        delete[] text_info;
        delete[] working_directory;
        delete[] program_name;
	for (int i = 0; i < n_associated_files; i ++)
	{
		delete[] associated_files[i];
	}
	delete[] associated_files;
}

//////////////////////////////////////////////////////////////////////////////////////////
// Resize the command line options

void CmdLine::resizeCmdOptions()
{
	// Create a larger set of cmd options
        int* temp_n_cmd_options = new int[n_master_switches + 1];
        CmdOption*** temp_cmd_options = new CmdOption**[n_master_switches + 1];

        // Copy the old information
        for (int i = 0; i < n_master_switches; i ++)
        {
                temp_n_cmd_options[i] = n_cmd_options[i];
                temp_cmd_options[i] = cmd_options[i];
        }

        // Point to the new cmd options
        delete[] n_cmd_options;
        delete[] cmd_options;
        n_cmd_options = temp_n_cmd_options;
        cmd_options = temp_cmd_options;

        n_cmd_options[n_master_switches] = 0;
        cmd_options[n_master_switches] = NULL;
        n_master_switches ++;
}

void CmdLine::resizeCmdOptions(int n, CmdOption* option)
{
	// Allocate a new option array to fit the new one
        CmdOption** temp_cmd_options = new CmdOption*[n_cmd_options[n] + 1];

        for (int i = 0; i < n_cmd_options[n]; i ++)
        {
                temp_cmd_options[i] = cmd_options[n][i];
        }

        delete[] cmd_options[n];
        cmd_options[n] = temp_cmd_options;

        cmd_options[n][n_cmd_options[n]] = option;
        n_cmd_options[n] ++;
}

//////////////////////////////////////////////////////////////////////////////////////////

void CmdLine::info(const char *text)
{
        delete[] text_info;

	text_info = new char[strlen(text)+1];
	strcpy(text_info, text);
}

//////////////////////////////////////////////////////////////////////////////////////////

void CmdLine::addCmdOption(CmdOption *option)
{
	if (option->isMasterSwitch())
	{
		resizeCmdOptions();
	}

        resizeCmdOptions(n_master_switches - 1, option);
}

void CmdLine::addMasterSwitch(const char *text)
{
	CmdOption* option = new CmdOption(text, "", "", false);
	option->isMasterSwitch(true);
	addCmdOption(option);
}

void CmdLine::addICmdOption(const char *name, int *ptr, int init_value, const char *help, bool save_it)
{
	addCmdOption(new IntCmdOption(name, ptr, init_value, help, save_it));
}

void CmdLine::addBCmdOption(const char *name, bool *ptr, bool init_value, const char *help, bool save_it)
{
	addCmdOption(new BoolCmdOption(name, ptr, init_value, help, save_it));
}

void CmdLine::addFCmdOption(const char *name, float *ptr, float init_value, const char *help, bool save_it)
{
	addCmdOption(new FloatCmdOption(name, ptr, init_value, help, save_it));
}

void CmdLine::addSCmdOption(const char *name, char **ptr, const char *init_value, const char *help, bool save_it)
{
	addCmdOption(new StringCmdOption(name, ptr, init_value, help, save_it));
}

void CmdLine::addDCmdOption(const char *name, double *ptr, double init_value, const char *help, bool save_it)
{
	addCmdOption(new DoubleCmdOption(name, ptr, init_value, help, save_it));
}

void CmdLine::addLLCmdOption(const char *name, long long *ptr, long long init_value, const char *help, bool save_it)
{
	addCmdOption(new LongLongCmdOption(name, ptr, init_value, help, save_it));
}

void CmdLine::addICmdArg(const char *name, int *ptr, const char *help, bool save_it)
{
	IntCmdOption *option = new IntCmdOption(name, ptr, 0, help, save_it);
	option->isArgument(true);
	addCmdOption(option);
}

void CmdLine::addBCmdArg(const char *name, bool *ptr, const char *help, bool save_it)
{
	BoolCmdOption *option = new BoolCmdOption(name, ptr, false, help, save_it);
	option->isArgument(true);
	addCmdOption(option);
}

void CmdLine::addFCmdArg(const char *name, float *ptr, const char *help, bool save_it)
{
	FloatCmdOption *option = new FloatCmdOption(name, ptr, 0.0f, help, save_it);
	option->isArgument(true);
	addCmdOption(option);
}

void CmdLine::addSCmdArg(const char *name, char **ptr, const char *help, bool save_it)
{
	StringCmdOption *option = new StringCmdOption(name, ptr, "", help, save_it);
	option->isArgument(true);
	addCmdOption(option);
}

void CmdLine::addDCmdArg(const char *name, double *ptr, const char *help, bool save_it)
{
	DoubleCmdOption *option = new DoubleCmdOption(name, ptr, 0., help, save_it);
	option->isArgument(true);
	addCmdOption(option);
}

void CmdLine::addLLCmdArg(const char *name, long long *ptr, const char *help, bool save_it)
{
	LongLongCmdOption *option = new LongLongCmdOption(name, ptr, 0, help, save_it);
	option->isArgument(true);
	addCmdOption(option);
}

void CmdLine::addText(const char *text)
{
	CmdOption *option = new CmdOption(text, "", "", false);
	option->isText(true);
	addCmdOption(option);
}

//////////////////////////////////////////////////////////////////////////////////////////

int CmdLine::read(int argc_, char **argv_)
{
	delete[] program_name;
	program_name = new char[strlen(argv_[0])+1];
	strcpy(program_name, argv_[0]);

	char** argv = argv_+1;
	int argc = argc_-1;

	// Look for help request and the Master Switch
	master_switch = 0;
	if (argc >= 1)
	{
		if ( ! (strcmp(argv[0], "-h") && strcmp(argv[0], "-help") && strcmp(argv[0], "--help")) )
			help();

		for (int i = 1; i < n_master_switches; i++)
		{
			if (cmd_options[i][0]->isCurrent(&argc, &argv))
			{
				master_switch = i;
				break;
			}
		}
	}

	CmdOption **cmd_options_ = cmd_options[master_switch];
	int n_cmd_options_ = n_cmd_options[master_switch];

	// Initialize the options.
	for (int i = 0; i < n_cmd_options_; i++)
		cmd_options_[i]->initValue();

	while (argc > 0)
	{
		// First, check the option.
		int current_option = -1;
		for (int i = 0; i < n_cmd_options_; i++)
		{
			if (cmd_options_[i]->isCurrent(&argc, &argv))
			{
				current_option = i;
				break;
			}
		}

		if (current_option >= 0)
		{
			if (cmd_options_[current_option]->is_setted)
				error("CmdLine: option %s is setted twice", cmd_options_[current_option]->name);
			cmd_options_[current_option]->read(&argc, &argv);
			cmd_options_[current_option]->is_setted = true;
		}
		else
		{
			// Check for arguments
			for (int i = 0; i < n_cmd_options_; i++)
			{
				if (cmd_options_[i]->isArgument() && (!cmd_options_[i]->is_setted))
				{
					current_option = i;
					break;
				}
			}

			if (current_option >= 0)
			{
				cmd_options_[current_option]->read(&argc, &argv);
				cmd_options_[current_option]->is_setted = true;
			}
			else
				error("CmdLine: parse error near <%s>. Too many arguments.", argv[0]);
		}
	}

	// Check for empty arguments
	for (int i = 0; i < n_cmd_options_; i++)
	{
		if (cmd_options_[i]->isArgument() && (!cmd_options_[i]->is_setted))
		{
			message("CmdLine: not enough arguments!\n");
			help();
			return -1;
		}
	}

	if (getBOption("write log") == true)
	{
		File log_file;
		if (log_file.open("cmd.log", "w") == true)
		{
                        writeLog(log_file, false);
		}
	}
	return master_switch;
}

//////////////////////////////////////////////////////////////////////////////////////////
// RhhAHha AH AHa hha hahaAH Ha ha ha (What is that ??!)

void CmdLine::help()
{
	if (text_info)
		print("%s\n", text_info);

	for (int master_switch_ = 0; master_switch_ < n_master_switches; master_switch_++)
	{
		int n_cmd_options_ = n_cmd_options[master_switch_];
		CmdOption **cmd_options_ = cmd_options[master_switch_];

		int n_real_options = 0;
		for (int i = 0; i < n_cmd_options_; i++)
		{
			if (cmd_options_[i]->isOption())
				n_real_options++;
		}

		if (master_switch_ == 0)
		{
			print("#\n");
			print("# usage: %s", program_name);
			if (n_real_options > 0)
				print(" [options]");
		}
		else
		{
			print("\n#\n");
			print("# or: %s %s", program_name, cmd_options_[0]->name);
			if (n_real_options > 0)
				print(" [options]");
		}

		for (int i = 0; i < n_cmd_options_; i++)
		{
			if (cmd_options_[i]->isArgument())
				print(" <%s>", cmd_options_[i]->name);
		}
		print("\n#\n");

		// Cherche la longueur max du param
		int long_max = 0;
		for (int i = 0; i < n_cmd_options_; i++)
		{
			int laurence = 0;
			if (cmd_options_[i]->isArgument())
				laurence = strlen(cmd_options_[i]->name)+2;

			if (cmd_options_[i]->isOption())
				laurence = strlen(cmd_options_[i]->name)+strlen(cmd_options_[i]->type_name)+1;

			if (long_max < laurence)
				long_max = laurence;
		}

		for (int i = 0; i < n_cmd_options_; i++)
		{
			int z = 0;
			if (cmd_options_[i]->isText())
			{
				z = -1;
				print("%s", cmd_options_[i]->name);
			}

			if (cmd_options_[i]->isArgument())
			{
				z = strlen(cmd_options_[i]->name)+2;
				print("  ");
				print("<%s>", cmd_options_[i]->name);
			}

			if (cmd_options_[i]->isOption())
			{
				z = strlen(cmd_options_[i]->name)+strlen(cmd_options_[i]->type_name)+1;
				print("  ");
				print("%s", cmd_options_[i]->name);
				print(" %s", cmd_options_[i]->type_name);
			}

			if (z >= 0)
			{
				for (int i = 0; i < long_max+1-z; i++)
					print(" ");
			}

			if ( cmd_options_[i]->isOption() || cmd_options_[i]->isArgument() )
				print("-> %s", cmd_options_[i]->help);

			if (cmd_options_[i]->isArgument())
				print(" (%s)", cmd_options_[i]->type_name);

			if (cmd_options_[i]->isOption())
			{
				File stdout_file;
				stdout_file.open(stdout);
				print(" ");
				cmd_options_[i]->printValue(stdout_file);
			}

			if (!cmd_options_[i]->isMasterSwitch())
				print("\n");
		}
	}
}

//////////////////////////////////////////////////////////////////////////////////////////

void CmdLine::setWorkingDirectory(const char* dirname)
{
	delete[] working_directory;
	working_directory = new char[strlen(dirname)+1];
	strcpy(working_directory, dirname);
}

//////////////////////////////////////////////////////////////////////////////////////////

char *CmdLine::getPath(const char *filename)
{
	// Resize the associated files
	char** temp_associated_files = new char*[n_associated_files + 1];
	for (int i = 0; i < n_associated_files; i ++)
	{
		temp_associated_files[i] = associated_files[i];
	}
	delete[] associated_files;
	associated_files = temp_associated_files;

	// Build the path
	char *path_ = new char[strlen(working_directory)+strlen(filename)+2];
	strcpy(path_, working_directory);
	strcat(path_, "/");
	strcat(path_, filename);

	// Copy the path in the associated files ?!
	associated_files[n_associated_files] = new char[strlen(filename)+1];
	strcpy(associated_files[n_associated_files], filename);
	n_associated_files ++;

	return path_;
}

//////////////////////////////////////////////////////////////////////////////////////////

bool CmdLine::saveFile(File& file) const
{
	if (master_switch < 0)
		error("CmdLine: nothing to save!");

	writeLog(file, true);

	file.taggedWrite(&master_switch, 1, "MASTER_SWITCH");
	CmdOption **cmd_options_ = cmd_options[master_switch];
	int n_cmd_options_ = n_cmd_options[master_switch];
	for (int i = 0; i < n_cmd_options_; i++)
	{
		if (cmd_options_[i]->save)
			cmd_options_[i]->saveFile(file);
	}

	// TODO: should check if it was correctly saved!
	return true;
}

//////////////////////////////////////////////////////////////////////////////////////////

void CmdLine::writeLog(File& file, bool write_associated_files) const
{
	// Header
	time_t time_ = time(NULL);
	file.printf("# Date: %s", ctime(&time_));
	file.printf("# Program: %s\n", program_name);
	if (master_switch < 0)
		file.printf("\n# CmdLine not read\n");
	if (master_switch == 0)
		file.printf("\n# Mode: default\n");
	if (master_switch > 0)
		file.printf("\n# Mode: <%s>\n", cmd_options[master_switch][0]->name);

	CmdOption **cmd_options_ = cmd_options[master_switch];
	int n_cmd_options_ = n_cmd_options[master_switch];

	// Cherche la longueur max du param
	int long_max = 0;
	for (int i = 0; i < n_cmd_options_; i++)
	{
		int z = 0;
		if (cmd_options_[i]->isArgument())
			z = strlen(cmd_options_[i]->name)+2;

		if (cmd_options_[i]->isOption())
			z = strlen(cmd_options_[i]->name)+strlen(cmd_options_[i]->type_name)+1;

		if (long_max < z)
			long_max = z;
	}

	file.printf("\n# Arguments:\n");
	for (int i = 0; i < n_cmd_options_; i++)
	{
		if (!cmd_options_[i]->isArgument())
			continue;

		int z = strlen(cmd_options_[i]->name)+2;
		file.printf("    ");
		file.printf("%s", cmd_options_[i]->name);

		if (z >= 0)
		{
			for (int i = 0; i < long_max+1-z; i++)
				file.printf(" ");
		}

		cmd_options_[i]->printValue(file);
		file.printf("\n");
	}

	file.printf("\n# Options:\n");
	for (int i = 0; i < n_cmd_options_; i++)
	{
		if (!cmd_options_[i]->isOption())
			continue;

		int z = strlen(cmd_options_[i]->name)+2;
		if (cmd_options_[i]->is_setted)
			file.printf(" *  ");
		else
			file.printf("    ");

		file.printf("%s", cmd_options_[i]->name);

		if (z >= 0)
		{
			for (int i = 0; i < long_max+1-z; i++)
				file.printf(" ");
		}

		cmd_options_[i]->printValue(file);
		file.printf("\n");
	}

	if (write_associated_files)
	{
		file.printf("\n# Associated files:\n");
		for (int i = 0; i < n_associated_files; i++)
			file.printf("    %s\n", associated_files[i]);
	}

	file.printf("\n<#>\n\n");
}

//////////////////////////////////////////////////////////////////////////////////////////

bool CmdLine::loadFile(File& file)
{
	// Skip the header
	int header_end = 0;
	while ( (header_end != 3) && (!file.eof()) )
	{
		char c;
		file.scanf("%c", &c);
		if (c == '<')
			header_end = 1;
		else
		{
			if (c == '#')
			{
				if (header_end == 1)
					header_end = 2;
				else
					header_end = 0;
			}
			else
			{
				if (c == '>')
				{
					if (header_end == 2)
					{
						header_end = 3;
						// the return-lines
						file.scanf("%c", &c);
						file.scanf("%c", &c);
					}
					else
						header_end = 0;
				}
				else
					header_end = 0;
			}
		}
	}

	if (header_end != 3)
		error("CmdLine: cannot find the end of the header!");

	//////////////////

	int old_master_switch;
	file.taggedRead(&old_master_switch, 1, "MASTER_SWITCH");
	CmdOption **cmd_options_ = cmd_options[old_master_switch];
	int n_cmd_options_ = n_cmd_options[old_master_switch];
	for (int i = 0; i < n_cmd_options_; i++)
	{
		if (cmd_options_[i]->save)
			cmd_options_[i]->loadFile(file);
	}

	// TODO: should check if it was correctly loaded!
	return true;
}

//////////////////////////////////////////////////////////////////////////////////////////

}
