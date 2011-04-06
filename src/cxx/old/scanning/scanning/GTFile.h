#ifndef _TORCH5SPRO_GT_FILE_H_
#define _TORCH5SPRO_GT_FILE_H_

#include "core/File.h"
#include "ip/vision.h"

namespace Torch
{
	class sPoint2D;

	/*
		NB: Labels are case insensitive!
		NB: By default all the labels are <undefined>,
			so in the specialized classes just put labels for the interest points!
		NB: For compound labels (e.g. "nose tip") use "_" as e.g. "nose_tip"!
	*/
	class GTFile : public Object
	{
	public:
		// Constructor
		GTFile(int n_points_ = 0);

		// Destructor
		virtual ~GTFile();

		// Load points from some file
		virtual bool		load(File* file) = 0;

		// Get the name of the GT file format
		virtual const char*	getName() = 0;

		// Check if some label is defined
		bool			hasLabel(const char* label) const;

		// Get the index for some label (or -1, if the label does not exist)
		int			getIndex(const char* label) const;

		// Access functions (if <index> is invalid, NULL/0 will be returned)
		int			getNPoints() const { return m_n_points; }
		const sPoint2D*		getPoints() const { return m_points; }
		const sPoint2D*		getPoint(int index) const;
		const sPoint2D*		getPoint(const char* label) const;
		const char*		getLabel(int index) const;

	protected:

		// Set a label at some point index
		bool			setLabel(int index, const char* label);

		///////////////////////////////////////////////////////////////////
		// Attributes

		int			m_n_points;
		sPoint2D*		m_points;		// Ground truth points
		char**			m_labels;		// Labels for each point
	};

}

#endif
