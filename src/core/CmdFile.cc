#include "CmdFile.h"
#include "File.h"

//#include <iostream.h>
//#include <fstream.h>

namespace Torch {

#define CMD_FILE_INT    0
#define CMD_FILE_SWITCH 1
#define CMD_FILE_STRING 2
#define CMD_FILE_DOUBLE  3

const char *cmd_file_type_str[4] = {"<int>", "", "<string>", "<double>"};

CmdFile::CmdFile()
{
  	n_cmd_options = 0;
  	n_cmd_params = 0;
  	cmd_options = NULL;
  	text_info = NULL;

  	strc = 0;
  	str = NULL;
}

void CmdFile::info(const char *text)
{
  	if(text_info)
  	{
    		free(text_info);
    		text_info = NULL;
  	}

  	text_info = (char *)malloc(strlen(text)+1);
  	strcpy(text_info, text);
}

void CmdFile::addCmdOption(const char *name, void *ptr, int type, const char *help, int status)
{
  	cmd_options = (CmdFileOption *)realloc((void *)cmd_options, (n_cmd_options+1)*sizeof(CmdFileOption));

  	CmdFileOption *optr = cmd_options+n_cmd_options;

  	optr->name = (char *)malloc(strlen(name)+1);
  	optr->help = (char *)malloc(strlen(help)+1);

  	strcpy(optr->name, name);
  	strcpy(optr->help, help);
  	optr->ptr = ptr;
  	optr->type = type;
  	optr->status = status;
  	optr->first_ptr = false;

  	if(status == CMD_FILE_PARAMS)
    		n_cmd_params++;

  	n_cmd_options++;
}

void CmdFile::addICmdOption(const char *name, int *ptr, int initvalue, const char *help)
{
  	*ptr = initvalue;
  	addCmdOption(name, (void *)ptr, CMD_FILE_INT, help, CMD_FILE_OPTION);
}

void CmdFile::addBCmdOption(const char *name, bool *ptr, bool initvalue, const char *help)
{
  	*ptr = initvalue;
  	addCmdOption(name, (void *)ptr, CMD_FILE_SWITCH, help, CMD_FILE_OPTION);
}

void CmdFile::addDCmdOption(const char *name, double *ptr, double initvalue, const char *help)
{
  	*ptr = initvalue;
  	addCmdOption(name, (void *)ptr, CMD_FILE_DOUBLE, help, CMD_FILE_OPTION);
}

void CmdFile::addSCmdOption(const char *name, char **ptr, const char *initvalue, const char *help)
{
  	*ptr = (char *)malloc(strlen(initvalue)+1);

  	strcpy(*ptr, initvalue);
  	addCmdOption(name, (void *)ptr, CMD_FILE_STRING, help, CMD_FILE_OPTION);
}

void CmdFile::addICmd(const char *name, int *ptr, const char *help)
{
  	addCmdOption(name, (void *)ptr, CMD_FILE_INT, help, CMD_FILE_PARAMS);
}

void CmdFile::addBCmd(const char *name, bool *ptr, const char *help)
{
  	addCmdOption(name, (void *)ptr, CMD_FILE_SWITCH, help, CMD_FILE_PARAMS);
}

void CmdFile::addDCmd(const char *name, double *ptr, const char *help)
{
  	addCmdOption(name, (void *)ptr, CMD_FILE_DOUBLE, help, CMD_FILE_PARAMS);
}

void CmdFile::addSCmd(const char *name, char **ptr, const char *help)
{
  	*ptr = NULL;
  	addCmdOption(name, (void *)ptr, CMD_FILE_STRING, help, CMD_FILE_PARAMS);
}

void CmdFile::addText(const char *text)
{
  	addCmdOption(text, NULL, 0, "", CMD_FILE_TEXT);
}

void CmdFile::printCmdOption(CmdFileOption *ptro)
{
  	char **ptr_s;

  	switch(ptro->type)
  	{
    	case CMD_FILE_INT:
      		print(" [%d]", *((int *)ptro->ptr));
      		break;
    	case CMD_FILE_DOUBLE:
      		print(" [%g]", *((double *)ptro->ptr));
      		break;
    	case CMD_FILE_STRING:
      		ptr_s = (char **)ptro->ptr;
      		print(" [%s]", *ptr_s);
      		break;
  	}
}

void CmdFile::setCmdOption(int argc, char **argv, int *current, CmdFileOption *ptro)
{
  	int current_ = *current;
  	int *ptr_i;
  	bool *ptr_b;
  	double *ptr_r;
  	char **ptr_s;

  	if(ptro->status == CMD_FILE_OPTION)
  	{
    		if( (current_ >= argc) && (ptro->type != CMD_FILE_SWITCH) )
      			error("CmdFile: cannot correctly set option <%s>", ptro->name);
  	}
  	else
  	{
    		if( (current_ >= argc) && (ptro->type != CMD_FILE_SWITCH) )
      			error("CmdFile: cannot correctly argument <%s>", ptro->name);
  	}

  	switch(ptro->type)
  	{
    	case CMD_FILE_INT:
      		ptr_i  = (int *)ptro->ptr;
      		*ptr_i = atoi(argv[current_++]);
      		break;
    	case CMD_FILE_SWITCH:
      		ptr_b  = (bool *)ptro->ptr;
		if(strcmp(argv[current_], "true") == 0) *ptr_b = true;
		else if(strcmp(argv[current_], "false") == 0) *ptr_b = false;
		else
		{
			warning("CmdFile: Incorrect bool value <%s> set to false", argv[current_]);

			*ptr_b = false;
		}
		current_++;
      		break;
    	case CMD_FILE_DOUBLE:
      		ptr_r  = (double *)ptro->ptr;
      		*ptr_r = (double)atof(argv[current_++]);
      		break;
    	case CMD_FILE_STRING:
		ptr_s  = (char **)ptro->ptr;
		free(*ptr_s);
		*ptr_s = (char *)malloc(strlen(argv[current_])+1);
		strcpy(*ptr_s, argv[current_++]);
      		break;
    	default:
      		error("CmdFile: wrong format");
  	}

  	*current = current_;
}

void CmdFile::setHrefCmdOption(int argc, char **argv, int *current, CmdFileOption *ptro)
{
  	int current_ = *current;
  	int *ptr_i;
  	bool *ptr_b;
  	double *ptr_r;
  	char **ptr_s;

  	if(ptro->status == CMD_FILE_OPTION)
  	{
    		if( (current_ >= argc) && (ptro->type != CMD_FILE_SWITCH) )
      			error("CmdFile: cannot correctly set option <%s>", ptro->name);
  	}
  	else
  	{
    		if( (current_ >= argc) && (ptro->type != CMD_FILE_SWITCH) )
      			error("CmdFile: cannot correctly argument <%s>", ptro->name);
  	}

  	switch(ptro->type)
  	{
    	case CMD_FILE_INT:
      		ptr_i  = (int *)ptro->ptr;
      		*ptr_i = atoi(argv[current_++]);
      		break;
    	case CMD_FILE_SWITCH:
      		ptr_b  = (bool *)ptro->ptr;
		if(strcmp(argv[current_], "true") == 0) *ptr_b = true;
		else if(strcmp(argv[current_], "false") == 0) *ptr_b = false;
		else
		{
			warning("CmdFile: Incorrect bool value <%s> set to false", argv[current_-1]);

			*ptr_b = false;
		}
		current_++;
      		break;
    	case CMD_FILE_DOUBLE:
      		ptr_r  = (double *)ptro->ptr;
      		*ptr_r = (double)atof(argv[current_++]);
      		break;
    	case CMD_FILE_STRING:
      		ptr_s  = (char **)ptro->ptr;
      		if(*ptr_s != NULL)
			free(*ptr_s);
		*ptr_s = (char *)malloc(strlen(argv[current_])+1);
      		strcpy(*ptr_s, argv[current_++]);
      		break;
    	default:
      		error("CmdFile: wrong format");
  	}

  	*current = current_;
}

void CmdFile::help()
{
  	if(text_info)
    		print("\n#### %s ####\n", text_info);

  	// Cherche la longueur max du param
  	int long_max = 0;
  	int z = 0;
  	for(int i = 0; i < n_cmd_options; i++)
  	{
    		switch(cmd_options[i].status)
    		{
      		case CMD_FILE_OPTION:
        		z = strlen(cmd_options[i].name)+strlen(cmd_file_type_str[cmd_options[i].type])+1;
        		break;
      		case CMD_FILE_PARAMS:
        		z = strlen(cmd_options[i].name)+2;
        		break;
    		}

    		if(long_max < z)
      			long_max = z;
  	}

  	for(int i = 0; i < n_cmd_options; i++)
  	{
    		switch(cmd_options[i].status)
    		{
      		case CMD_FILE_TEXT:
        		z = -1;
        		print("%s", cmd_options[i].name);
        		break;
      		case CMD_FILE_PARAMS:
      		case CMD_FILE_OPTION:
        		z = strlen(cmd_options[i].name)+strlen(cmd_file_type_str[cmd_options[i].type])+1;
        		print("  ");
        		print("%s", cmd_options[i].name);
        		print(" %s", cmd_file_type_str[cmd_options[i].type]);
        		break;
    		}

    		if(z >= 0)
    		{
      			for(int i = 0; i < long_max+1-z; i++)
        		print(" ");
    		}

    		if(cmd_options[i].status != CMD_FILE_TEXT)
      			print("-> %s", cmd_options[i].help);

    		switch(cmd_options[i].status)
    		{
      		case CMD_FILE_OPTION:
        		printCmdOption(&cmd_options[i]);
       			break;
      		case CMD_FILE_PARAMS:
        		print(" (%s)", cmd_file_type_str[cmd_options[i].type]);
        		break;
    		}
    		print("\n");
  	}

//  	exit(0);
}

CmdFile::~CmdFile()
{
  	char **ptr_s;
  	char **ptr;
  	int *ptr_i;
  	bool *ptr_b;
  	double *ptr_r;

//  	if(text_info) print("\n#### %s ####\n", text_info);

  	// Delete substrings
  	for(int i = 0; i < n_cmd_options; i++)
		if(cmd_options[i].status == CMD_FILE_HREF_PARAMS)
			if(cmd_options[i].type == CMD_FILE_STRING)
		  	{
				ptr = (char **)cmd_options[i].ptr;
			 	free(*ptr);
		  	}

  	for(int i = 0; i < n_cmd_options; i++)
  	{
    		free(cmd_options[i].name);

	 	if(cmd_options[i].status == CMD_FILE_HREF_PARAMS)
		{
		  	if(cmd_options[i].first_ptr == true)
			{
				switch(cmd_options[i].type)
				{
				case CMD_FILE_STRING:
					ptr_s = (char **)cmd_options[i].ptr;
					free(ptr_s);

					break;

				case CMD_FILE_INT:
					ptr_i = (int *)cmd_options[i].ptr;
					free(ptr_i);

					break;

				case CMD_FILE_SWITCH:
					ptr_b = (bool *)cmd_options[i].ptr;
					free(ptr_b);
					break;

				case CMD_FILE_DOUBLE:
					ptr_r = (double *)cmd_options[i].ptr;
					free(ptr_r);
					break;
				}
			}
		}
	 	else
		{
		  	if(cmd_options[i].type == CMD_FILE_STRING)
			{
				ptr = (char **)cmd_options[i].ptr;
				free(*ptr);
			}
		}

    		if(cmd_options[i].status != CMD_FILE_TEXT)
			free(cmd_options[i].help);
  	}

  	free(cmd_options);
  	free(text_info);

  	for(int i = 0 ; i < strc ; i++)
		free(str[i]);

  	free(str);
}

void CmdFile::read(char *filename, bool check_everything)
{
  	int ld;
  	int the_opt, current;
  	char dummy[500];

  	File ParamFile;
  	if (ParamFile.open(filename, "r") == false)
  	{
  		// ?!
  	}

  	//if(!ParamFile) error("CmdFile: Impossible to open file %s", filename);

  	strc = 0;
  	str = NULL;

  	ParamFile.scanf("%s", dummy);
	//print("[%s]\n", dummy);

  	if(dummy[0] != '#')
	{
		ld = strlen(dummy);
		str = (char **) malloc(sizeof(char*));
		str[strc] = (char *) malloc(ld + 1);
		strcpy(str[strc], dummy);
		strc++;

		ParamFile.scanf("%s", dummy);
		//print("[%s]\n", dummy);
		ld = strlen(dummy) ;
		str = (char **) realloc((void *)str, (strc + 1) * sizeof(char*));
		str[strc] = (char *) malloc(ld + 1);
		strcpy(str[strc], dummy);
		strc++;
	}
  	else
	{
		ParamFile.scanf("%s", dummy);
		//print("[%s]\n", dummy);
	}

  	while(!ParamFile.eof())
	{
		ParamFile.scanf("%s", dummy);
		//print("[%s]\n", dummy);
		if(ParamFile.eof()) break;
		if(dummy[0] != '#')
		{
			 ld = strlen(dummy);
			 if(strc == 0)
			 	str = (char **) malloc(sizeof(char*));
			 else
			 	str = (char **) realloc((void *)str, (strc + 1) * sizeof(char*));

			 str[strc] = (char *) malloc(ld + 1);
			 strcpy(str[strc], dummy);
			 strc++;

			 ParamFile.scanf("%s", dummy);
			 //print("[%s]\n", dummy);
			 if(ParamFile.eof()) break;
			 ld = strlen(dummy);
			 str = (char **) realloc((void *)str, (strc + 1) * sizeof(char*));
			 str[strc] = (char *) malloc(ld + 1);
			 strcpy(str[strc], dummy);

			 strc++;
		}
		else
		{
			 ParamFile.scanf("%s", dummy);
			 //print("[%s]\n", dummy);
		}
	}

	ParamFile.close();

//  	print("\n%d :\n", strc);
//  	for(int i = 0 ; i < strc ; i++)
//  	 	print(" [%s]\n", str[i]);

//  	message("Check all params there");

  	// Check all Params
  	for(int j = 0; j < n_cmd_options ; j++)
	{
		if(cmd_options[j].status == CMD_FILE_PARAMS)
		{
			the_opt = -1;

			current = 0;

			while(current < strc)
			{
				if(strcmp(cmd_options[j].name, str[current]) == 0)
				{
					the_opt = j;
					break;
				}

				current += 2;
			}

			if(the_opt == -1) error("CmdFile: parameter <%s> not found", cmd_options[j].name);
		}
	}


  	if(check_everything)
	{
//		message("Check everything is option or param");

		current = 0;
		while(current < strc)
		{
			// Look for an option
			the_opt = -1;
			for(int j = 0; j < n_cmd_options; j++)
			{
				if((cmd_options[j].status == CMD_FILE_OPTION) || (cmd_options[j].status == CMD_FILE_PARAMS))
				{
					if(strcmp(cmd_options[j].name, str[current]) == 0)
					{
						the_opt = j;
						break;
					}
				}
			}

			if(the_opt == -1) error("CmdFile: <%s> option or parameter incorrect", str[current]);

			current += 2;
		}
	}

//  	message("Load options and params");

  	current = 0;
  	while(current < strc)
	{
		// Look for an option or a parameter
		the_opt = -1;
		for(int j = 0; j < n_cmd_options; j++)
		{
			if((cmd_options[j].status == CMD_FILE_OPTION) || (cmd_options[j].status == CMD_FILE_PARAMS))
			{
				if(strcmp(cmd_options[j].name, str[current]) == 0)
				{
					the_opt = j;
					break;
				}
			}
		}

		if(the_opt != -1) // This is an option or a parameter because everything was check
		{
			current++;

			setCmdOption(strc, str, &current, &cmd_options[the_opt]);
		}
		else
		{
			if(check_everything) warning("Ignoring <%s>", str[current]);

			current += 2;
		}
	}
}

void CmdFile::addHrefCmdOption(const char *name, void **ptr, int type, const char *help, int status, int max_)
{
  	char s[200];

  	for(int i = 0 ; i < max_ ; i++)
	{
		cmd_options = (CmdFileOption *)realloc((void *)cmd_options, (n_cmd_options+1)*sizeof(CmdFileOption));

		CmdFileOption *optr = cmd_options+n_cmd_options;

		sprintf(s, name, i);

		optr->name = (char *)malloc(strlen(s)+1);
		optr->help = (char *)malloc(strlen(help)+1);
		strcpy(optr->name, s);
		strcpy(optr->help, help);
		optr->ptr = &ptr[i];
		optr->type = type;
		optr->status = status;
		if(i == 0) optr->first_ptr = true;
		else optr->first_ptr = false;

		if((status == CMD_FILE_PARAMS) || (status == CMD_FILE_HREF_PARAMS)) n_cmd_params++;

		n_cmd_options++;
	}
}

void CmdFile::addHrefICmdOption(const char *name, int **ptr, int initvalue, const char *help, int max_)
{
  	*ptr = NULL;
  	int *p;

  	p = (int *) malloc(max_ * sizeof(int));

  	*ptr = p;

  	for(int i = 0 ; i < max_ ; i++) p[i] = initvalue;

  	addHrefCmdOption(name, (void **)p, CMD_FILE_INT, help, CMD_FILE_HREF_OPTION, max_);
}

void CmdFile::addHrefBCmdOption(const char *name, bool **ptr, bool initvalue, const char *help, int max_)
{
  	*ptr = NULL;
  	bool *p;

  	p = (bool *)malloc(max_ * sizeof(bool));

  	*ptr = p;

  	for(int i = 0 ; i < max_ ; i++) p[i] = initvalue;

  	addHrefCmdOption(name, (void **)p, CMD_FILE_SWITCH, help, CMD_FILE_HREF_OPTION, max_);
}

void CmdFile::addHrefDCmdOption(const char *name, double **ptr, double initvalue, const char *help, int max_)
{
  	*ptr = NULL;
  	double *p;

  	p = (double *)malloc(max_ * sizeof(double));

  	*ptr = p;

  	for(int i = 0 ; i < max_ ; i++) p[i] = initvalue;

  	addHrefCmdOption(name, (void **)p, CMD_FILE_DOUBLE, help, CMD_FILE_HREF_OPTION, max_);
}

void CmdFile::addHrefSCmdOption(const char *name, char ***ptr, const char *initvalue, const char *help, int max_)
{
  	*ptr = NULL;
  	char **p;

  	p = (char **) malloc(max_ * sizeof(char*));

  	*ptr = p;

  	for(int i = 0 ; i < max_ ; i++)
	{
		p[i] = (char *) malloc((strlen(initvalue) + 1));

		strcpy(p[i], initvalue);
	}

  	addHrefCmdOption(name, (void **)p, CMD_FILE_STRING, help, CMD_FILE_HREF_OPTION, max_);
}

void CmdFile::addHrefICmd(const char *name, int **ptr, const char *help, int max_)
{
  	*ptr = NULL;
  	int *p;

  	p = (int *) malloc(max_ * sizeof(int));

  	*ptr = p;

  	for(int i = 0 ; i < max_ ; i++) p[i] = 0;

  	addHrefCmdOption(name, (void **)p, CMD_FILE_INT, help, CMD_FILE_HREF_PARAMS, max_);
}

void CmdFile::addHrefBCmd(const char *name, bool **ptr, const char *help, int max_)
{
  	*ptr = NULL;
  	bool *p;

  	p = (bool *) malloc(max_ * sizeof(bool));

  	*ptr = p;

  	for(int i = 0 ; i < max_ ; i++) p[i] = false;

  	addHrefCmdOption(name, (void **)p, CMD_FILE_SWITCH, help, CMD_FILE_HREF_PARAMS, max_);
}

void CmdFile::addHrefDCmd(const char *name, double **ptr, const char *help, int max_)
{
  	*ptr = NULL;
  	double *p;

  	p = (double *) malloc(max_ * sizeof(double));

  	*ptr = p;

  	for(int i = 0 ; i < max_ ; i++) p[i] = 0.0;

  	addHrefCmdOption(name, (void **)p, CMD_FILE_DOUBLE, help, CMD_FILE_HREF_PARAMS, max_);
}

void CmdFile::addHrefSCmd(const char *name, char ***ptr, const char *help, int max_)
{
  	*ptr = NULL;
  	char **p;

  	p = (char **) malloc(max_ * sizeof(char*));

  	*ptr = p;

  	for(int i = 0 ; i < max_ ; i++) p[i] = NULL;

  	addHrefCmdOption(name, (void **)p, CMD_FILE_STRING, help, CMD_FILE_HREF_PARAMS, max_);
}

void CmdFile::readHref()
{
  	int the_opt, current;

  	if(strc == 0) error("CmdFile: Parameters not loaded");

//  	print("\n%d :\n", strc);
//  	for(int i = 0 ; i < strc ; i++)
//  		print(" [%s]\n", str[i]);

//  	message("Check all params there");

  	// Check all Params
  	for(int j = 0; j < n_cmd_options ; j++)
	{
		if(cmd_options[j].status == CMD_FILE_HREF_PARAMS)
		{
			the_opt = -1;

			current = 0;

			while(current < strc)
			{
				if(strcmp(cmd_options[j].name, str[current]) == 0)
				{
					the_opt = j;
					break;
				}

				current += 2;
			}

			if(the_opt == -1) error("CmdFile: parameter <%s> not found", cmd_options[j].name);
		}
	}


//  	message("Check everything is option or param");

  	current = 0;
  	while(current < strc)
	{
		// Look for an option
		the_opt = -1;
		for(int j = 0; j < n_cmd_options; j++)
		{
			if((cmd_options[j].status == CMD_FILE_OPTION)
				 || (cmd_options[j].status == CMD_FILE_PARAMS)
				 || (cmd_options[j].status == CMD_FILE_HREF_OPTION)
				 || (cmd_options[j].status == CMD_FILE_HREF_PARAMS))
			{
				if(strcmp(cmd_options[j].name, str[current]) == 0)
				{
					the_opt = j;
					break;
				}
			}
		}

		if(the_opt == -1) error("CmdFile: <%s> option or parameter incorrect", str[current]);

		current += 2;
	}

//  	message("Load options and params");

  	current = 0;
  	while(current < strc)
	{
		// Look for an option or a parameter
		the_opt = -1;
		for(int j = 0; j < n_cmd_options; j++)
		{
			if((cmd_options[j].status == CMD_FILE_HREF_OPTION) || (cmd_options[j].status == CMD_FILE_HREF_PARAMS))
			{
				if(strcmp(cmd_options[j].name, str[current]) == 0)
				{
					the_opt = j;
					break;
				}
			}
		}

		if(the_opt != -1) // This is an option or a parameter because everything was check
		{
			current++;

			setHrefCmdOption(strc, str, &current, &cmd_options[the_opt]);
		}
		else
		{
			warning("Ignoring <%s>", str[current]);

			current += 2;
		}
	}
}

} // End namespace

