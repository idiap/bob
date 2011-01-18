#ifndef _TORCH5SPRO_PROFILE_ENC_GT_FILE_H_
#define _TORCH5SPRO_PROFILE_ENC_GT_FILE_H_

#include "scanning/GTFile.h"

namespace Torch
{
	/**
		Labels: leye_center, nose_tip, chin

	*/
	class profileEyeNoseChinGTFile : public GTFile
	{
	public:
		// Constructor
		profileEyeNoseChinGTFile();

		// Destructor
		virtual ~profileEyeNoseChinGTFile();

		// Load points from some file
		virtual bool		load(File* file);

		// Get the name of the GT file format
		virtual const char*	getName();
	};

}

#endif
