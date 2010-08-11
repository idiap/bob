#include "core/CmdOption.h"
#include "core/File.h"

namespace Torch {

CmdOption::CmdOption(const char *name_, const char *type_name_, const char *help_, bool save_)
{
	name = new char[strlen(name_)+1];
	strcpy(name, name_);

	type_name = new char[strlen(type_name_)+1];
	strcpy(type_name, type_name_);

	help = new char[strlen(help_)+1];
	strcpy(help, help_);

	save = save_;
	is_setted = false;

	is_option = true;
	is_argument = false;
	is_text = false;
	is_master_switch = false;
}

bool CmdOption::isOption(bool set_)
{
	if (set_)
	{
		is_option = true;
		is_argument = false;
		is_text = false;
		is_master_switch = false;
	}
	return is_option;
}

bool CmdOption::isArgument(bool set_)
{
	if (set_)
	{
		is_option = false;
		is_argument = true;
		is_text = false;
		is_master_switch = false;
	}
	return is_argument;
}

bool CmdOption::isText(bool set_)
{
	if (set_)
	{
		is_option = false;
		is_argument = false;
		is_text = true;
		is_master_switch = false;
	}
	return is_text;
}

bool CmdOption::isMasterSwitch(bool set_)
{
	if (set_)
	{
		is_option = false;
		is_argument = false;
		is_text = false;
		is_master_switch = true;
	}
	return is_master_switch;
}

bool CmdOption::isCurrent(int *argc_, char ***argv_)
{
	if (!is_option && !is_master_switch)
		return false;

	if (strcmp((*argv_)[0], name))
		return false;
	else
	{
		(*argc_)--;
		(*argv_)++;
		return true;
	}
}

CmdOption::~CmdOption()
{
	delete[] name;
	delete[] type_name;
	delete[] help;
}

//-------------------------- int

IntCmdOption::IntCmdOption(const char *name_, int *ptr_, int init_value_, const char *help_, bool save_)
		: CmdOption(name_, "<int>", help_, save_)
{
	ptr = ptr_;
	init_value = init_value_;
}

void IntCmdOption::initValue()
{
	*ptr = init_value;
}

void IntCmdOption::printValue(File& file) const
{
	if (is_setted)
		file.printf("[%d]", *ptr);
	else
		file.printf("[%d]", init_value);
}


void IntCmdOption::read(int *argc_, char ***argv_)
{
	char **argv = *argv_;
	char *maryline;

	if (*argc_ == 0)
		error("IntCmdOption: cannot correctly set <%s>", name);

	*ptr = strtol(argv[0], &maryline, 10);
	if ( *maryline != '\0' )
		error("IntCmdOption: <%s> requires an integer", name);

	(*argc_)--;
	(*argv_)++;
}

bool IntCmdOption::loadFile(File& file)
{
	return file.taggedRead(ptr, 1, name) == 1;
}

bool IntCmdOption::saveFile(File& file) const
{
	return file.taggedWrite(ptr, 1, name) == 1;
}

IntCmdOption::~IntCmdOption()
{
}


//-------------------------- float

FloatCmdOption::FloatCmdOption(const char *name_, float *ptr_, float init_value_, const char *help_, bool save_)
		: CmdOption(name_, "<float>", help_, save_)
{
	ptr = ptr_;
	init_value = init_value_;
}

void FloatCmdOption::initValue()
{
	*ptr = init_value;
}

void FloatCmdOption::printValue(File& file) const
{
	if (is_setted)
		file.printf("[%g]", *ptr);
	else
		file.printf("[%g]", init_value);
}

void FloatCmdOption::read(int *argc_, char ***argv_)
{
	char **argv = *argv_;
	char *maryline;

	if (*argc_ == 0)
		error("FloatCmdOption: cannot correctly set <%s>", name);

	*ptr = strtod(argv[0], &maryline);
	if ( *maryline != '\0' )
		error("FloatCmdOption: <%s> requires a float", name);

	(*argc_)--;
	(*argv_)++;
}

bool FloatCmdOption::loadFile(File& file)
{
	return file.taggedRead(ptr, 1, name) == 1;
}

bool FloatCmdOption::saveFile(File& file) const
{
	return file.taggedWrite(ptr, 1, name) == 1;
}

FloatCmdOption::~FloatCmdOption()
{
}

//-------------------------- switch

BoolCmdOption::BoolCmdOption(const char *name_, bool *ptr_, bool init_value_, const char *help_, bool save_)
		: CmdOption(name_, "", help_, save_)
{
	ptr = ptr_;
	init_value = init_value_;
}

void BoolCmdOption::initValue()
{
	*ptr = init_value;
}

void BoolCmdOption::read(int *argc_, char ***argv_)
{
	*ptr = !(*ptr);
}

bool BoolCmdOption::loadFile(File& file)
{
	int melanie;
	const bool ret = file.taggedRead(&melanie, 1, name) == 1;
	*ptr = (melanie ? 1 : 0);
	return ret;
}

bool BoolCmdOption::saveFile(File& file) const
{
	int melanie = (*ptr ? 1 : 0);
	return file.taggedWrite(&melanie, 1, name) == 1;
}

BoolCmdOption::~BoolCmdOption()
{
}

//-------------------------- string

StringCmdOption::StringCmdOption(const char *name_, char **ptr_, const char *init_value_, const char *help_, bool save_)
		: CmdOption(name_, "<string>", help_, save_)
{
	*ptr_ = 0;
	ptr = ptr_;
	init_value = new char[strlen(init_value_)+1];
	strcpy(init_value, init_value_);
}

void StringCmdOption::initValue()
{
	*ptr = new char[strlen(init_value)+1];
	strcpy(*ptr, init_value);
}

void StringCmdOption::printValue(File& file) const
{
	if (is_setted)
		file.printf("[%s]", *ptr);
	else
		file.printf("[%s]", init_value);
}


void StringCmdOption::read(int *argc_, char ***argv_)
{
	char **argv = *argv_;

	if (*argc_ == 0)
		error("StringCmdOption: cannot correctly set <%s>", name);

	delete[] *ptr;
	*ptr = new char[strlen(argv[0])+1];
	strcpy(*ptr, argv[0]);

	(*argc_)--;
	(*argv_)++;
}

bool StringCmdOption::loadFile(File& file)
{
	int melanie;
	if (file.taggedRead(&melanie, 1, "SIZE") != 1)
	{
                return false;
        }
	*ptr = new char[melanie];
	if (file.taggedRead(*ptr,  melanie, name) != melanie)
	{
	        return false;
	}

	// TODO: should check if it was correctly loaded!
	return true;
}

bool StringCmdOption::saveFile(File& file) const
{
	int melanie = strlen(*ptr)+1;
	if (file.taggedWrite(&melanie, 1, "SIZE") != 1)
	{
	        return false;
	}
	if (file.taggedWrite(*ptr, melanie, name) != melanie)
	{
	        return false;
	}

	// TODO: should check if it was correctly saved!
	return true;
}

StringCmdOption::~StringCmdOption()
{
	delete[] *ptr;
        delete[] init_value;
        *ptr = 0;
}

//-------------------------- double

DoubleCmdOption::DoubleCmdOption(const char *name_, double *ptr_, double init_value_, const char *help_, bool save_)
		: CmdOption(name_, "<double>", help_, save_)
{
	ptr = ptr_;
	init_value = init_value_;
}

void DoubleCmdOption::initValue()
{
	*ptr = init_value;
}

void DoubleCmdOption::printValue(File& file) const
{
	if (is_setted)
		file.printf("[%g]", *ptr);
	else
		file.printf("[%g]", init_value);
}


void DoubleCmdOption::read(int *argc_, char ***argv_)
{
	char **argv = *argv_;
	char *maryline;

	if (*argc_ == 0)
		error("DoubleCmdOption: cannot correctly set <%s>", name);

	*ptr = strtod(argv[0], &maryline);
	if ( *maryline != '\0' )
		error("DoubleCmdOption: <%s> requires a double", name);

	(*argc_)--;
	(*argv_)++;
}

bool DoubleCmdOption::loadFile(File& file)
{
	return file.taggedRead(ptr, 1, name) == 1;
}

bool DoubleCmdOption::saveFile(File& file) const
{
	return file.taggedWrite(ptr, 1, name) == 1;
}

DoubleCmdOption::~DoubleCmdOption()
{
}


//-------------------------- long long

LongLongCmdOption::LongLongCmdOption(const char *name_, long long *ptr_, long long init_value_, const char *help_, bool save_)
		: CmdOption(name_, "<longlong>", help_, save_)
{
	ptr = ptr_;
	init_value = init_value_;
}

void LongLongCmdOption::initValue()
{
	*ptr = init_value;
}

void LongLongCmdOption::printValue(File& file) const
{
	if (is_setted)
		file.printf("[%lld]", *ptr);
	else
		file.printf("[%lld]", init_value);
}


void LongLongCmdOption::read(int *argc_, char ***argv_)
{
	char **argv = *argv_;
	char *maryline;

	if (*argc_ == 0)
		error("LongLongCmdOption: cannot correctly set <%s>", name);

	*ptr = strtol(argv[0], &maryline, 10);
	if ( *maryline != '\0' )
		error("LongLongCmdOption: <%s> requires an integer", name);

	(*argc_)--;
	(*argv_)++;
}

bool LongLongCmdOption::loadFile(File& file)
{
	return file.taggedRead(ptr, 1, name) == 1;
}

bool LongLongCmdOption::saveFile(File& file) const
{
	return file.taggedWrite(ptr, 1, name) == 1;
}

LongLongCmdOption::~LongLongCmdOption()
{
}


}
