#ifndef _TORCH5SPRO_EYECORNER_GT_FILE_H_
#define _TORCH5SPRO_EYECORNER_GT_FILE_H_

#include "GTFile.h"

namespace Torch
{
	/**
		Labels: leye_ocorner, leye_icorner
			reye_ocorner, reye_icorner
			leye_center, reye_center

	*/
	class eyecornerGTFile : public GTFile
	{
	public:
		// Constructor
		eyecornerGTFile();

		// Destructor
		virtual ~eyecornerGTFile();

		// Load points from some file
		virtual bool		load(File* file);

		// Get the name of the GT file format
		virtual const char*	getName();
	};

}

#endif
