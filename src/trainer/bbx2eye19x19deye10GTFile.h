#ifndef _TORCH5SPRO_BBX2EYE19X19DEYE10_GT_FILE_H_
#define _TORCH5SPRO_BBX2EYE19X19DEYE10_GT_FILE_H_

#include "GTFile.h"
#include "scanning/Pattern.h"

namespace Torch
{
	/**
		Labels: leye_center, reye_center

	*/
	class bbx2eye19x19deye10_GTFile : public GTFile
	{
	public:
		// Constructor
		bbx2eye19x19deye10_GTFile();

		// Destructor
		virtual ~bbx2eye19x19deye10_GTFile();

		// Load points from some file
		virtual bool		load(File* file);

    /**
     * Load points already in memory
     * @param p The pattern containing the bounding box you want to use
     *
     * @return Always true...
     */
    virtual bool    load(const Torch::Pattern& p);

		// Get the name of the GT file format
		virtual const char*	getName();
	};

}

#endif
