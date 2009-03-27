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

		// Load an image
		virtual bool		load(File* file);

		// Get the name of the label of a point
		virtual const char*	getLabel(int);

		// Get the name of the GT file format
		virtual const char*	getName();
	};

}

#endif
