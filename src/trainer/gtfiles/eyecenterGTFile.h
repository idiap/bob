#ifndef _TORCH5SPRO_EYECENTER_GT_FILE_H_
#define _TORCH5SPRO_EYECENTER_GT_FILE_H_

#include "GTFile.h"

namespace Torch
{
	/**
	*/
	class eyecenterGTFile : public GTFile
	{
	public:
		// Constructor
		eyecenterGTFile();

		// Destructor
		virtual ~eyecenterGTFile();

		// Load points from some file
		virtual bool		load(File* file);

		// Get the name of the GT file format
		virtual const char*	getName();
	};

}

#endif
