#ifndef _TORCH5SPRO_FILE_LIST_CMD_OPTION_H_
#define _TORCH5SPRO_FILE_LIST_CMD_OPTION_H_

#include "CmdOption.h"

namespace Torch {

        ////////////////////////////////////////////////////////////////////////////////////
        /** This class take a file name in the command line,
            and reads a list of files contained in this
            file.

            In fact, there is a special case: it checks first
            if "-one_file" the current argument on the command
            line. If true, then it reads the next argument which
            will be the only file in the list.

            @author Ronan Collobert (collober@idiap.ch)
            @see CmdLine
        */
        ////////////////////////////////////////////////////////////////////////////////////
        class FileListCmdOption : public CmdOption
        {
        public:
                /// Constructor
                FileListCmdOption(const char *name_, const char *help_="", bool save_=false);

                /// Destructor
                ~FileListCmdOption();

                /// Initialize the value of the option. - overriden
		//virtual void initValue();

		/// If #is_setted# is true, print the current value, else the init value. - overriden
		//virtual void printValue(File& file) const;

		/// Read the option on the command line. - overriden
		virtual void read(int *argc_, char ***argv_);

		/// Loading/Saving the content from files (<em>not the options</em>) - overriden
		virtual bool loadFile(File& file);
		virtual bool saveFile(File& file) const;

		////////////////////////////////////////////////////////////////////////////////////

	public:

		////////////////////////////////////////////////////////////////////////////////////
		// Attributes

		/// Contains the file names after reading the command line.
                char **file_names;

                /// Number of files that have been read.
                int n_files;
        };

}

#endif
