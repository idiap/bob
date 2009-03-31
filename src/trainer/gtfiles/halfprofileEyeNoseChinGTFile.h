#ifndef _TORCH5SPRO_HALFPROFILE_ENC_GT_FILE_H_
#define _TORCH5SPRO_HALFPROFILE_ENC_GT_FILE_H_

#include "GTFile.h"

namespace Torch
{
	/**
	*/
	class halfprofileEyeNoseChinGTFile : public GTFile
	{
	public:
		// Constructor
		halfprofileEyeNoseChinGTFile();

		// Destructor
		virtual ~halfprofileEyeNoseChinGTFile();

		// Load points from some file
		virtual bool		load(File* file);

		// Get the name of the GT file format
		virtual const char*	getName();
	};

}

#endif
