#ifndef _TORCH5SPRO_GT_FILE_H_
#define _TORCH5SPRO_GT_FILE_H_

#include "File.h"
#include "vision.h"

namespace Torch
{
	class sPoint2D;

	/**
	*/
	class GTFile : public Object
	{
	public:
		// Constructor
		GTFile(int n_points_ = 0);

		// Destructor
		virtual ~GTFile();

		// Load an image
		virtual bool		load(File* file) = 0;

		//
		int			getNPoints() const { return m_n_points; }
		sPoint2D*		getPoints() const { return m_points; }

		// Get the name of the label of a point
		virtual char *		getLabel(int) = 0;

		// Get the name of the GT file format
		virtual char *		getName() = 0;

	protected:

		///////////////////////////////////////////////////////////////////
		// Attributes

		int			m_n_points;
		sPoint2D*		m_points;
	};

}

#endif
