#ifndef _TORCH5SPRO_CMD_LINE_H_
#define _TORCH5SPRO_CMD_LINE_H_

#include "Object.h"
#include "CmdOption.h"

namespace Torch
{
        ////////////////////////////////////////////////////////////////////////////////////
        /** This class provides a useful interface for the user,
            to easily read some arguments/options from the command-line.

            Note that here, we make a difference between:
            \begin{itemize}
              \item {\bf options} which are not required.
              \item {\bf arguments} which are required.
            \end{itemize}

            Options:
            \begin{tabular}{lcll}
              "write log"  &  bool  &  Should I output the cmd.log file ? & [true]
            \end{tabular}

            Parameters (name, type, default value, help):
                "write log"     bool    false    "Should I output the cmd.log file ?"

            @author Ronan Collobert (collober@idiap.ch)
            @see CmdOption
        */
        ////////////////////////////////////////////////////////////////////////////////////
        class CmdLine : public Object
        {
        public:
                // -----

                /// Constructor
                CmdLine();

		/// Destructor
                virtual ~CmdLine();

                /** Read the command-line.
                    Call this function {\bf after} adding options/arguments
                    that you need, with the help of the following functions.
                */
                int read(int argc_, char **argv_);

                /** Print the help.
                    Call this function {\bf after} adding options/arguments
                    that you need, with the help of the following functions.
                */
                void help();

                //-----

                /** Functions for adding options.
                    The calling order of the following functions will
                    define the text order associated when you will call #help()#.

                    Add an option (Int, Bool, Float, String).
                    \begin{itemize}
                      \item #name# the name of the option (must be unique).
                      \item #ptr# is the pointer on the optional variable.
                      \item #init_value# is the initialization value.
                      \item #help# is the help text for this option.
                    \end{itemize}

                    The option will be setted to #value# in the command-line
                    by printing "#name# #value#"
                */
                ///
                void addICmdOption(const char *name, int *ptr, int init_value, const char *help="", bool save_it=false);
                ///
                void addBCmdOption(const char *name, bool *ptr, bool init_value, const char *help="", bool save_it=false);
                ///
                void addFCmdOption(const char *name, float *ptr, float init_value, const char *help="", bool save_it=false);
                ///
                void addSCmdOption(const char *name, char **ptr, const char *init_value, const char *help="", bool save_it=false);
                ///
                void addDCmdOption(const char *name, double *ptr, double init_value, const char *help="", bool save_it=false);
                ///
                void addLLCmdOption(const char *name, long long *ptr, long long init_value, const char *help="", bool save_it=false);

                /** Functions for adding an argument.
                    The argument will be setted to #value# in the command-line
                    by writting "#value#" {\bf after} all the options.
                    If there are N arguments, you have to write
                    "#value1# #value2# #value3# ... #valueN#" to set them in the
                    command-line.
                */
                ///
                void addICmdArg(const char *name, int *ptr, const char *help="", bool save_it=false);
                ///
                void addBCmdArg(const char *name, bool *ptr, const char *help="", bool save_it=false);
                ///
                void addFCmdArg(const char *name, float *ptr, const char *help="", bool save_it=false);
                ///
                void addSCmdArg(const char *name, char **ptr, const char *help="", bool save_it=false);
                ///
                void addDCmdArg(const char *name, double *ptr, const char *help="", bool save_it=false);
                ///
                void addLLCmdArg(const char *name, long long *ptr, const char *help="", bool save_it=false);

                /// Add a text line in the help message.
                void addText(const char *text);

                /// Add a text at the beginnig of the help.
                void info(const char *text);

                /** Add a master switch.
                    It creates an another type of command line.
                    If the #text# is the first argument of the user command line,
                    only the options corresponding to this new command line will be considered.
                */
                void addMasterSwitch(const char *text);

                /** Set the working directory.
                    Use it with #getPath()# and #getXFile()#.
                */
                void setWorkingDirectory(const char* dirname);

                /** Get a full path.
                    It adds the #working_directory# before the #filename#.
                    This path will be deleted by CmdLine. */
                char *getPath(const char *filename);

                /// Loading/Saving the content from files (<em>not the options</em>) - overriden
                virtual bool loadFile(File& file);
                virtual bool saveFile(File& file) const;

                //-----

                /** Add an option to the command line. Use this method
                    if the wrappers that are provided are not sufficient.
                */
                void addCmdOption(CmdOption *option);

                /** Write a log in #file#.
                    If desired, the associated files can be printed.
                */
                void writeLog(File& file, bool write_associated_files) const;

                ////////////////////////////////////////////////////////////////////////////////////

	private:

		// Deallocate memory
		void		cleanup();

                // Resize the command line options
                void            resizeCmdOptions();
                void            resizeCmdOptions(int index_master_switch, CmdOption* option);

		////////////////////////////////////////////////////////////////////////////////////
		// Attributes

		char *program_name;
		char *text_info;
                char *working_directory;
                char **associated_files;
                int n_associated_files;

                int n_master_switches;
                int master_switch;
                int *n_cmd_options;
                CmdOption ***cmd_options;
        };
}

#endif
