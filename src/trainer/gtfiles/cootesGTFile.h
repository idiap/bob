#ifndef _TORCH5SPRO_COOTES_GT_FILE_H_
#define _TORCH5SPRO_COOTES_GT_FILE_H_

#include "GTFile.h"

namespace Torch
{
	/** 
	*/
	class cootesGTFile : public GTFile
	{
	public:
		// Constructor
		cootesGTFile();

		// Destructor
		virtual ~cootesGTFile();

		// Load an image
		virtual bool		load(File* file);

		// Get the name of the label of a point
		virtual const char*	getLabel(int);

		// Get the name of the GT file format
		virtual const char*	getName();
	};

}

#endif
