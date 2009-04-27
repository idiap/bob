#ifndef _TORCH5SPRO_BANCA_GT_FILE_H_
#define _TORCH5SPRO_BANCA_GT_FILE_H_

#include "GTFile.h"

namespace Torch
{
	/**
		Labels: leye_center, reye_center

	*/
	class bancaGTFile : public GTFile
	{
	public:
		// Constructor
		bancaGTFile();

		// Destructor
		virtual ~bancaGTFile();

		// Load points from some file
		virtual bool		load(File* file);

		// Get the name of the GT file format
		virtual const char*	getName();
	};
}

#endif
