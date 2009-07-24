#ifndef _TORCH5SPRO_FILE_LIST_H_
#define _TORCH5SPRO_FILE_LIST_H_

#include "Object.h"

namespace Torch {

        ////////////////////////////////////////////////////////////////////////////////////
        /** This class take a file name
            and reads a list of files contained in this
            file.
        */
        ////////////////////////////////////////////////////////////////////////////////////
        class FileList : public Object
        {
        public:
                /// Constructor
                FileList(const char *name_);

                /// Destructor
                ~FileList();

		////////////////////////////////////////////////////////////////////////////////////

	public:

		////////////////////////////////////////////////////////////////////////////////////
		// Attributes

		/// Contains the file names after reading the command line.
                char **file_names;

                /// Number of files that have been read.
                int n_files;
	private:
                char **allocated_file_names;
        };

}

#endif
