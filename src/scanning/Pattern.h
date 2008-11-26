#ifndef _TORCHVISION_SCANNING_PATTERN_H_
#define _TORCHVISION_SCANNING_PATTERN_H_

namespace Torch
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::Pattern
	//	- represent a pattern in 4D (position + scale + model confidence)
	//		scanning space!
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	struct Pattern
	{
		// Constructor
		Pattern(	int x = 0, int y = 0, int w = 0, int h = 0,
				float confidence = 0.0f)
			: 	m_x(x), m_y(y), m_w(w), m_h(h),
				m_confidence(confidence)
		{
		}

		// Copy from another pattern
		void		copy(const Pattern& other)
		{
			m_x = other.m_x;
			m_y = other.m_y;
			m_w = other.m_w;
			m_h = other.m_h;
			m_confidence = other.m_confidence;
		}

		/////////////////////////////////////////////////////////////////
		// Attributes

		// Position: location + scale (top, left, width, height)
		int		m_x, m_y;
		int		m_w, m_h;

		// Model confidence
		float		m_confidence;
	};

	/////////////////////////////////////////////////////////////////
	// Torch::PatternList
	//	- smart implementation of a set of patterns, using an array
	//		of fixed size arrays of patterns
	//	- can be resized and iterated
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////

	class PatternList
	{
	public:

		// Constructor
		PatternList(	bool trackBestPatterns = false,
				int maxBestPatterns = 128);

		// Destructor
		~PatternList();

		// Add a new pattern - returns a reference to the stored pattern
		Pattern&		add(const Pattern& pattern);

		// Invalidates all stored patterns (they are not deallocated, but ready to be used again)
		void			clear();

		// Deallocate all stored patterns
		void			deallocate();

		// Change if the best patterns are stored and their maximum number
		bool			resetBestPatterns(bool trackBestPatterns, int maxBestPatterns = 128);

		// Access functions
		bool			isEmpty() const;
		int			size() const;
		int			capacity() const;
		const Pattern&		get(int index) const;
		int			getMaxNoBest() const;
		int			getNoBest() const;
		const Pattern&		getBest(int index) const;

	private:

		//////////////////////////////////////////////////////////////////
		// Node in the pattern list

		struct Node
		{
			// Constructor
			Node();

			// Destructor
			~Node();

			// Add a new pattern - returns a pattern
			// (presuming that there is enough allocated memory)
			Pattern&	add(const Pattern& pattern);

			///////////////////////////////////////////////////////////
			// Attributes

			Pattern*	m_patterns;
			int		m_n_used;
			int		m_n_allocated;
		};

		//////////////////////////////////////////////////////////////////

		// One node in the list will use pattern arrays of this size
		static const int	PatternArraySize = 1024;

		// Node size increment
		static const int 	NodeArraySizeIncrement = 32;

		// Get the node/pattern index from a global index
		static int		getNodeIndex(int index)
		{
			return index / PatternArraySize;
		}
		static int		getPatternIndex(int index)
		{
			return index % PatternArraySize;
		}

		// Add a new pattern in the ordered list of the best patterns
		void			addBest(const Pattern& pattern, int index_pattern);

		// Finds the position where some pattern should be inserted in order to keep the list of
		//	the best ones as sorted
		int			findBest(const Pattern& pattern, int index1, int index2) const;

		//////////////////////////////////////////////////////////////////
		// Attributes

		// Array of nodes
		Node**			m_nodes;
		int			m_idx_last_node;
		int			m_n_allocated_nodes;

		// Best patterns
		bool			m_best_activated;	// Flag: should keep track of the best patterns
		int*			m_best_indexes;		// Best N patterns kept as indexes
		int			m_best_size;		// Actual number of stored best patterns
		int			m_best_max_size;	// Maximum number of best patterns to keep track of

		// Some statistics
		int			m_n_used_patterns;	// Total number of used patterns
		int			m_n_allocated_patterns;	// Total number of allocated patterns
	};

	/////////////////////////////////////////////////////////////////
	// Torch::PatternSpace
	//	- 4D pattern space representation (position + scale + pattern confidence),
	//		used for fast retrieving the patterns close to some specific point
	//		either in space or scale
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////

	class PatternSpace
	{
	public:
		// Constructor
		PatternSpace();

		// Destructor
		~PatternSpace();

		// Reset to a new context:
		//	- images of different sizes
		//	- new model threshold
		//	- number of best points to keep track of
		// <clear> is called!
		bool			reset(int image_w, int image_h, float model_threshold);
		bool			reset(int n_best);

		// Delete all stored patterns (but it's not deallocating the memory)
		void			clear();

		// Add a new candidate pattern (sub-window coordinates + model confidence)
		void			add(const Pattern& pattern);

		// Check if some scanning point is already stored
		bool			hasPoint(int sw_x, int sw_y, int sw_w, int sw_h);

		/////////////////////////////////////////////////////////////////////////
		// Access functions

		bool			isEmpty() const { return m_patterns.isEmpty(); }
		int			size() const { return m_patterns.size(); }
		int			getMaxNoBest() const { return m_patterns.getMaxNoBest(); }
		int			getNoBest() const { return m_patterns.getNoBest(); }
		const Pattern&		getBest(int index) const { return m_patterns.getBest(index); }

		/////////////////////////////////////////////////////////////////////////

	private:

		/////////////////////////////////////////////////////////////////////////

		// Normalize&Scale some model confidence
		int			normScaleConfidence(float confidence) const
		{
			return (int)(0.5f + (confidence - m_model_threshold) * 100.0f);
		}

		// Rescale&UnNormalize some model confidence
		float			unNormScaleConfidence(int ns_confidence) const
		{
			return m_model_threshold + ns_confidence * 0.01f;
		}

		// Deallocate the allocated tables and pattern list
		void			deallocateTables();
		void			deallocatePatterns();

		/////////////////////////////////////////////////////////////////////////

		/////////////////////////////////////////////////////////////////////////
		// Attributes

		// Image size (gives the used portion of the allocated tables)
		int			m_image_w, m_image_h;

		// Model threshold: used for normalizing and scalling the model output
		float			m_model_threshold;

		// All patterns (also keeps track of the best patterns)
		PatternList		m_patterns;

		// Tables: normalized summed confidence, used sub-window positions
		int**			m_table_confidence;	// [m_image_w x m_image_h]
		unsigned char**		m_table_used_xy;	// [m_image_w x m_image_h]
	};
}

#endif

