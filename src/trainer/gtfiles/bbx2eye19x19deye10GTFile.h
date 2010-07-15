#ifndef _TORCH5SPRO_BBX2EYE19X19DEYE10_GT_FILE_H_
#define _TORCH5SPRO_BBX2EYE19X19DEYE10_GT_FILE_H_

#include "GTFile.h"

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
     * @param x The top left ordinate of the bounding box
     * @param y The top left abcissa of the bounding box
     * @param w The width of the bounding box
     * @param w The height of the bounding box
     *
     * @return Always true...
     */
    virtual bool    load(short x, short y, short w, short h);

		// Get the name of the GT file format
		virtual const char*	getName();
	};

}

#endif
