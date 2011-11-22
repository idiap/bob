/**
 * @file cxx/old/scanning/scanning/ContextExplorer.h
 * @date Wed Apr 6 14:49:40 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef _TORCHVISION_SCANNING_CONTEXT_EXPLORER_H_
#define _TORCHVISION_SCANNING_CONTEXT_EXPLORER_H_

#include "scanning/MSExplorer.h"		// <ContextExplorer> is a <MSExplorer>
#include "scanning/ContextMachine.h"

namespace Torch
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::ContextExplorer
	//	- searches the 4D scanning space using a greedy method based on
	//		a context-based model to remove false alarms
	//		and drive iteratively the detections to better locations
	//
	//	- PARAMETERS (name, type, default value, description):
	//		"ctx_type"	int	1	"0 - full context, 1 - axis context"
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class ContextExplorer : public MSExplorer
	{
	public:

		// Running mode
		enum Mode
		{
			Scanning,	// Context-based greedy scanning
			Profiling	// Initialize the scanning - used for generating contexts
		};

		// Context type
		enum ContextType
		{
			Full,
			Axis
		};

		// Constructor
		ContextExplorer(Mode mode = Scanning);

		// Destructor
		virtual ~ContextExplorer();

		/////////////////////////////////////////////////////////////////
		// Process functions

		// HOW TO USE (see Scanner):
		// --------------------------------
		// init(image_w, image_h)
		// ... setScaleXXX as wanted
		// preprocess(image)
		// for each ROI
		//	init (ROI)
		// 	process ()
		// --------------------------------

		// Process the image (check for pattern's sub-windows)
		virtual bool		process();

		// Build the context for some SW
		bool			buildSWContext(int sw_x, int sw_y, int sw_w, int sw_h);
		bool			buildSWContext(const Pattern& pattern);

		// Set the context classifier
		bool			setContextModel(const char* filename);

		// Access functions
		void			setMode(Mode mode) { m_mode = mode; }
		Mode 			getMode() const { return m_mode; }
		const unsigned char*	getContextFlags() const { return m_ctx_flags; }
		const double*		getContextScores() const { return m_ctx_scores; }
		int			getContextOx(int index) const { return m_ctx_ox[index]; }
		int			getContextOy(int index) const { return m_ctx_oy[index]; }
		int			getContextOs(int index) const { return m_ctx_os[index]; }
		int			getContextSize() const { return m_ctx_size; }

		/////////////////////////////////////////////////////////////////

	protected:

		// Initialize the context-based scanning
		virtual bool		initContext();

		// Test a subwindow
		virtual void		testSW(	int sw_x, int sw_y, int sw_w, int sw_h,
							unsigned char& flag, double& score);

		/// called when some option was changed
		virtual void		optionChanged(const char* name);

		/////////////////////////////////////////////////////////////////
		// Attributes

		// Current working mode
		Mode 			m_mode;		// Scanning (MS + context model), Profiling (MS)

		// Context model, type, buffers
		ContextMachine		m_ctx_model;
		ContextType		m_ctx_type;
		int			m_ctx_size;
		unsigned char*		m_ctx_flags;
		double*			m_ctx_scores;
		int*			m_ctx_ox;
		int*			m_ctx_oy;
		int*			m_ctx_os;
		PatternMerger*		m_ctx_sw_merger;
	};
}

#endif
