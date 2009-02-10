#ifndef _TORCH5SPRO_CMD_OPTION_H_
#define _TORCH5SPRO_CMD_OPTION_H_

#include "Object.h"

namespace Torch
{
	class File;

	////////////////////////////////////////////////////////////////////////////////////
	/** This class defines an option for the command line.
	    If you need special command line arguments/options,
	    you have to create a new children of this class.

	    @author Ronan Collobert (collober@idiap.ch)
	    @see CmdLine
	*/
	////////////////////////////////////////////////////////////////////////////////////
	class CmdOption : public Object
	{
	public:
		/// Constructor
		CmdOption(const char *name_, const char *type_name_, const char *help_="", bool save_=false);

		/// Destructor
		virtual ~CmdOption();

		////////////////////////////////////////////////////////////////////////////////////

		/// Initialize the value of the option.
		virtual void initValue() {};

		/// If #is_setted# is true, print the current value, else the init value.
		virtual void printValue(File& file) const {}

		/** Read the option on the command line.
		    argv_ and argc_ have to point of the next
		    option after that.
		*/
		virtual void read(int *argc_, char ***argv_) {};

		/// Loading/Saving the content from files (\emph{not the options})
		virtual bool loadFile(File& file) { return true; }
		virtual bool saveFile(File& file) const { return true; }

		////////////////////////////////////////////////////////////////////////////////////

		/* Return true if the option is on the command line.
		   Decrements argc_ and increment argv_ if true.
		*/
		bool isCurrent(int *argc_, char ***argv_);

		/** Returns true if it's an optional argument.
		    If #set_# is true, set it to an optional argument.
		*/
		bool isOption(bool set_=false);

		/** Returns true if it's a required argument.
		    If #set_# is true, set it to a required argument.
		*/
		bool isArgument(bool set_=false);

		/** Returns true if it's just text to be displayed in the command line.
		    If #set_# is true, set it to text mode.
		*/
		bool isText(bool set_=false);

		/** Returns true if it's a master switch.
		    If #set_# is true, set it to a master switch.
		*/
		bool isMasterSwitch(bool set_=false);

		////////////////////////////////////////////////////////////////////////////////////

	private:

		////////////////////////////////////////////////////////////////////////////////////
		// Attributes

		// Special flags.
		bool is_option;
		bool is_argument;
		bool is_text;
		bool is_master_switch;

	public:

		/// Name of the option.
		char *name;

		/// Type name of the option.
		char *type_name;

		/** An help string.
		    Cannot be NULL.
		*/
		char *help;

		/** True is the option has to be saved
		    when saving the command line.
		*/
		bool save;

		/** True is the option has been setted after
		    reading the command-line.
		*/
		bool is_setted;
	};

	////////////////////////////////////////////////////////////////////////////////////
	/** This class defines a integer command-line option.

	    @author Ronan Collobert (collober@idiap.ch)
	    @see CmdLine
	*/
	////////////////////////////////////////////////////////////////////////////////////
	class IntCmdOption : public CmdOption
	{
	public:
		/// Constructor
		IntCmdOption(const char *name_, int *ptr_, int init_value_, const char *help_="", bool save_=false);

		/// Destructor
		~IntCmdOption();

		////////////////////////////////////////////////////////////////////////////////////

		/// Initialize the value of the option. - overriden
		virtual void initValue();

		/// If #is_setted# is true, print the current value, else the init value. - overriden
		virtual void printValue(File& file) const;

		/// Read the option on the command line. - overriden
		virtual void read(int *argc_, char ***argv_);

		/// Loading/Saving the content from files (\emph{not the options}) - overriden
		virtual bool loadFile(File& file);
		virtual bool saveFile(File& file) const;

		////////////////////////////////////////////////////////////////////////////////////

	public:

		////////////////////////////////////////////////////////////////////////////////////
		// Attributes

		int *ptr;
		int init_value;
	};

	////////////////////////////////////////////////////////////////////////////////////
	/** This class defines a float command-line option.

	    @author Ronan Collobert (collober@idiap.ch)
	    @see CmdLine
	*/
	////////////////////////////////////////////////////////////////////////////////////
	class FloatCmdOption : public CmdOption
	{
	public:
		/// Constructor
		FloatCmdOption(const char *name_, float *ptr_, float init_value_, const char *help_="", bool save_=false);

		/// Destructor
		~FloatCmdOption();

		/// Initialize the value of the option. - overriden
		virtual void initValue();

		/// If #is_setted# is true, print the current value, else the init value. - overriden
		virtual void printValue(File& file) const;

		/// Read the option on the command line. - overriden
		virtual void read(int *argc_, char ***argv_);

		/// Loading/Saving the content from files (\emph{not the options}) - overriden
		virtual bool loadFile(File& file);
		virtual bool saveFile(File& file) const;

		////////////////////////////////////////////////////////////////////////////////////

	public:

		////////////////////////////////////////////////////////////////////////////////////
		// Attributes

		float *ptr;
		float init_value;
	};

	////////////////////////////////////////////////////////////////////////////////////
	/** This class defines a double command-line option.

	    @author Ronan Collobert (collober@idiap.ch)
	    @see CmdLine
	*/
	////////////////////////////////////////////////////////////////////////////////////
	class DoubleCmdOption : public CmdOption
	{
	public:
		/// Constructor
		DoubleCmdOption(const char *name_, double *ptr_, double init_value_, const char *help_="", bool save_=false);

		/// Destructor
		~DoubleCmdOption();

		/// Initialize the value of the option. - overriden
		virtual void initValue();

		/// If #is_setted# is true, print the current value, else the init value. - overriden
		virtual void printValue(File& file) const;

		/// Read the option on the command line. - overriden
		virtual void read(int *argc_, char ***argv_);

		/// Loading/Saving the content from files (\emph{not the options}) - overriden
		virtual bool loadFile(File& file);
		virtual bool saveFile(File& file) const;

		////////////////////////////////////////////////////////////////////////////////////

	public:

		////////////////////////////////////////////////////////////////////////////////////
		// Attributes

		double *ptr;
		double init_value;
	};

	////////////////////////////////////////////////////////////////////////////////////
	/** This class defines a bool command-line option.

	    @author Ronan Collobert (collober@idiap.ch)
	    @see CmdLine
	*/
	////////////////////////////////////////////////////////////////////////////////////
	class BoolCmdOption : public CmdOption
	{
	public:
		/// Constructor
		BoolCmdOption(const char *name_, bool *ptr_, bool init_value_, const char *help_="", bool save_=false);

		/// Destructor
		~BoolCmdOption();

		////////////////////////////////////////////////////////////////////////////////////

		/// Initialize the value of the option. - overriden
		virtual void initValue();

		/// If #is_setted# is true, print the current value, else the init value. - overriden
		//virtual void printValue(File& file) const;

		/// Read the option on the command line. - overriden
		virtual void read(int *argc_, char ***argv_);

		/// Loading/Saving the content from files (\emph{not the options}) - overriden
		virtual bool loadFile(File& file);
		virtual bool saveFile(File& file) const;

		////////////////////////////////////////////////////////////////////////////////////

	public:

		////////////////////////////////////////////////////////////////////////////////////
		// Attributes

		bool *ptr;
		bool init_value;
	};

	////////////////////////////////////////////////////////////////////////////////////
	/** This class defines a string command-line option.

	    @author Ronan Collobert (collober@idiap.ch)
	    @see CmdLine
	*/
	////////////////////////////////////////////////////////////////////////////////////
	class StringCmdOption : public CmdOption
	{
	public:

		/// Constructor
		StringCmdOption(const char *name_, char **ptr_, const char *init_value_, const char *help_="", bool save_=false);

		/// Destructor
		~StringCmdOption();

		////////////////////////////////////////////////////////////////////////////////////

		/// Initialize the value of the option. - overriden
		virtual void initValue();

		/// If #is_setted# is true, print the current value, else the init value. - overriden
		virtual void printValue(File& file) const;

		/// Read the option on the command line. - overriden
		virtual void read(int *argc_, char ***argv_);

		/// Loading/Saving the content from files (\emph{not the options}) - overriden
		virtual bool loadFile(File& file);
		virtual bool saveFile(File& file) const;

		////////////////////////////////////////////////////////////////////////////////////

	public:

		////////////////////////////////////////////////////////////////////////////////////
		// Attributes

		char **ptr;
		char *init_value;
	};

	////////////////////////////////////////////////////////////////////////////////////
	/** This class defines a long long integer command-line option.

	    @author Sebastien Marcel (marcel@idiap.ch)
	    @see CmdLine
	*/
	////////////////////////////////////////////////////////////////////////////////////
	class LongLongCmdOption : public CmdOption
	{
	public:
		/// Constructor
		LongLongCmdOption(const char *name_, long long *ptr_, long long init_value_, const char *help_="", bool save_=false);

		/// Destructor
		~LongLongCmdOption();

		////////////////////////////////////////////////////////////////////////////////////

		/// Initialize the value of the option. - overriden
		virtual void initValue();

		/// If #is_setted# is true, print the current value, else the init value. - overriden
		virtual void printValue(File& file) const;

		/// Read the option on the command line. - overriden
		virtual void read(int *argc_, char ***argv_);

		/// Loading/Saving the content from files (\emph{not the options}) - overriden
		virtual bool loadFile(File& file);
		virtual bool saveFile(File& file) const;

		////////////////////////////////////////////////////////////////////////////////////

	public:

		////////////////////////////////////////////////////////////////////////////////////
		// Attributes

		long long *ptr;
		long long init_value;
	};
}

#endif
