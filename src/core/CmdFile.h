#ifndef _TORCH5SPRO_CMD_FILE_H_
#define _TORCH5SPRO_CMD_FILE_H_

#include "Object.h"

namespace Torch {

// Internal
#define CMD_FILE_TEXT   0
#define CMD_FILE_OPTION 1
#define CMD_FILE_PARAMS 2
#define CMD_FILE_HREF_OPTION 1
#define CMD_FILE_HREF_PARAMS 2

	typedef struct CmdFileOption_
	{
		char *name;
		char *help;
		void *ptr;
		int type;
		int status;
		bool first_ptr;
	} CmdFileOption;
	//

	/** This class is designed to read parameters from a file

	    This is a clone of CmdLine but for files.

	    @author Sebastien Marcel (marcel@idiap.ch)
	    @version 2.0
	    \date
	    @since 2.0
	*/
	class CmdFile : public Object
	{
	public:
		int n_cmd_options;
		int n_cmd_params;
		CmdFileOption *cmd_options;
		char *text_info;

		int strc;
		char **str;

		//-----

		///
		CmdFile();

		///
		void read(const char *, bool = true);

		/// Write the current option values to some file (in the required format for ::read)
		bool write(File& file) const;
		bool write(const char* filename) const;

		///
		void readHref();

		///
		void help() const;

		//-----

		///
		void addICmdOption(const char *name, int *ptr, int initvalue, const char *help="");

		///
		void addBCmdOption(const char *name, bool *ptr, bool initvalue, const char *help="");

		///
		void addDCmdOption(const char *name, double *ptr, double initvalue, const char *help="");

		///
		void addSCmdOption(const char *name, char **ptr, const char *initvalue, const char *help="");

		///
		void addICmd(const char *name, int *ptr, const char *help="");

		///
		void addBCmd(const char *name, bool *ptr, const char *help="");

		///
		void addDCmd(const char *name, double *ptr, const char *help="");

		///
		void addSCmd(const char *name, char **ptr, const char *help="");

		//-----

		///
		void addHrefICmdOption(const char *name, int **ptr, int initvalue, const char *help="", int = 1);

		///
		void addHrefBCmdOption(const char *name, bool **ptr, bool initvalue, const char *help="", int = 1);

		///
		void addHrefDCmdOption(const char *name, double **ptr, double initvalue, const char *help="", int = 1);

		///
		void addHrefSCmdOption(const char *name, char ***ptr, const char *initvalue, const char *help="", int = 1);

		///
		void addHrefICmd(const char *name, int **ptr, const char *help="", int = 1);

		///
		void addHrefBCmd(const char *name, bool **ptr, const char *help="", int = 1);

		///
		void addHrefDCmd(const char *name, double **ptr, const char *help="", int = 1);

		///
		void addHrefSCmd(const char *name, char ***ptr, const char *help="", int = 1);

		/// Add a text line in the help message.
		void addText(const char *text);

		/// Add a text at the beginnig of the help.
		void info(const char *text);

		//-----

		///
		void addCmdOption(const char *name, void *ptr, int type, const char *help="", int status=CMD_FILE_OPTION);

		///
		void addHrefCmdOption(const char *name, void **ptr, int type, const char *help="", int status=CMD_FILE_OPTION, int = 1);

		///
		void setCmdOption(int argc, char **argv, int *current, CmdFileOption *ptro);

		///
		void setHrefCmdOption(int argc, char **argv, int *current, CmdFileOption *ptro);

		///
		void printCmdOption(CmdFileOption *ptro) const;

		///
		virtual ~CmdFile();
	};
}

#endif
