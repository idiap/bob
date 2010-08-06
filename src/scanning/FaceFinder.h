#ifndef _TORCHVISION_FACE_FINDER_H_
#define _TORCHVISION_FACE_FINDER_H_

#include "core/Object.h"			// <FaceFinder> is a <Torch::Object>
#include "Pattern.h"			// detected patterns
#include "core/CmdFile.h"

namespace Torch
{
	class Image;
	class Explorer;
	class ScaleExplorer;
	class Scanner;
	class spCoreChain;

	/////////////////////////////////////////////////////////////////////////
	// Torch::FaceFinder
	//	- groups together the main scanning objects
	//		to make the face detector easier to use
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class FaceFinder : public Torch::Object
	{
	public:

		struct Params;

		// Constructor
		FaceFinder(Params* params);
		FaceFinder(const char* filename = 0);

		// Destructor
		virtual ~FaceFinder();

		// Change the scanning parameters
		virtual bool			reset(Params* params);
		virtual bool			reset(const char* filename);

		// Process some image to scan for patterns
		virtual bool	 		process(const Image& image);

		/////////////////////////////////////////////////////////////////
		// Access functions

		const PatternList&		getPatterns() const;
		const Scanner&			getScanner() const;
		Scanner&			getScanner();

		/////////////////////////////////////////////////////////////////////////
		// Gathers all the parameters needed to scan for faces
		struct Params
		{
			// Constructor
			Params(const char* filename = 0);

			// Destructor
			virtual ~Params();

			// Load/save the configuration to/from a CmdFile-like file
			virtual bool		load(const char* filename);
			virtual bool		save(const char* filename) const;

			// Display current parameter values
			virtual bool		print() const;

			// Attributes
			CmdFile			cmd_file;		// Load parameters using this object

			char* 			f_model;		// Face model filename
			double			f_thres;		// Face model threshold
			bool			use_f_thres;		// USe the face model threshold

			int			explorer_type;		// Explorer: 0 - Pyramid, 1 - Multiscale, 2 - Context, 3 - Track Context
			int			scale_explorer_type;    // ScaleExplorer: 0 - Exhaustive, 1 - Spiral, 2 - Random
			int			min_patt_w, max_patt_w; // Min/max pattern width/height
			int			min_patt_h, max_patt_h;
			double 			dx, dy, ds;               // Scanning precision factors
			bool 			stop_at_first_detection;   // Flag
			bool 			start_with_large_scales;   // Flag

			int 			random_nsamples;	// Random scale explorer specific

			bool 			prune_use_mean;          // Prune sub-windows using the mean
			bool 			prune_use_stdev;         // Prune sub-windows using the stdev
			double 			prune_min_mean;          // Prune using mean: min value
			double 			prune_max_mean;          // Prune using mean: max value
			double 			prune_min_stdev;         // Prune using stdev: min value
			double 			prune_max_stdev;         // Prune using stdev: max value

			bool 			prep_ii;		// Preprocessing: Compute integral image
			bool 			prep_hlbp;		// Preprocessing: HLBP: compute LBP4R bitmaps

			int 			select_type;		// Merge type: 0 - Overlap, 1 - MeanShift, 2 - NoMerge
			int			select_overlap_type;	// Overlap: 0 - Average, 1 - Confidence Weighted, 2 - Maximum Confidence
			bool			select_overlap_iterative;// Overlap: Iterative/One step
			int 			select_overlap_min_surf;// Overlap: Minimum surface overlap to merge

			char* 			context_model;		// Context model to remove FAs, used by ContextExplorer
			int 			context_type;		// Context type (0 - Full, 1 - Axis)

			bool 			verbose;		// General verbose flag
		};

	protected:

		/////////////////////////////////////////////////////////////////
		// Attributes

		Torch::Scanner* 		m_scanner;
		Torch::spCoreChain*		m_sp_prep;
		Torch::Explorer*		m_explorer;
		Torch::ScaleExplorer*		m_scale_explorer;
	};
}

#endif
