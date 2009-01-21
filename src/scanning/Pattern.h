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
		Pattern(	short x = 0, short y = 0, short w = 0, short h = 0,
				double confidence = 0.0,
				short activation = 1)
			: 	m_x(x), m_y(y), m_w(w), m_h(h),
				m_confidence(confidence),
				m_activation(activation)
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
			m_activation = other.m_activation;
		}

		// Returns the percentage of the overlapping area of intersection with another one
		int		getOverlap(const Pattern& other, bool ignoreInclusion = true) const;

		// Compute the center of the SW
		int		getCenterX() const { return m_x + m_w / 2; }
		int		getCenterY() const { return m_y + m_h / 2; }

		/////////////////////////////////////////////////////////////////
		// Attributes

		// Position: location + scale (top, left, width, height)
		short		m_x, m_y;
		short		m_w, m_h;

		// Model confidence
		double		m_confidence;

		// Model activation (if merged, from how many SWs)
		short		m_activation;
	};

	/////////////////////////////////////////////////////////////////////////
	// Torch::Scanning::PatternMerger
	//	- generic class for merging a list of patterns
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class PatternMerger
	{
	public:
		// Constructor
		PatternMerger() { }

		// Destructor
		virtual ~PatternMerger() { }

		// Reset the merging
		virtual void		reset() = 0;

		// Accumulate some pattern (from the merging list?!)
		virtual void		add(const Pattern& pattern) = 0;

		// Copy the merged result to the given pattern
		virtual void		merge(Pattern& pattern) const = 0;
	};

	class AveragePatternMerger : public PatternMerger
	{
	public:
		// Constructor
		AveragePatternMerger();

		// Destructor
		virtual ~AveragePatternMerger();

		// Reset the merging
		virtual void		reset();

		// Accumulate some pattern (from the merging list?!)
		virtual void		add(const Pattern& pattern);

		// Copy the merged result to the given pattern
		virtual void		merge(Pattern& pattern) const;

	private:

		//////////////////////////////////////////////////////////////////
		// Attributes

		double			m_sum_cx, m_sum_cy;
		double			m_sum_w, m_sum_h;
		double			m_sum_confidence;
		int			m_count;
	};

	class ConfWeightedPatternMerger : public PatternMerger
	{
	public:
		// Constructor
		ConfWeightedPatternMerger();

		// Destructor
		virtual ~ConfWeightedPatternMerger();

		// Reset the merging
		virtual void		reset();

		// Accumulate some pattern (from the merging list?!)
		virtual void		add(const Pattern& pattern);

		// Copy the merged result to the given pattern
		virtual void		merge(Pattern& pattern) const;

	private:

		//////////////////////////////////////////////////////////////////
		// Attributes

		double			m_sum_cx, m_sum_cy;
		double			m_sum_w, m_sum_h;
		double			m_sum_confidence;
		int			m_count;
	};

	class MaxConfPatternMerger : public PatternMerger
	{
	public:
		// Constructor
		MaxConfPatternMerger();

		// Destructor
		virtual ~MaxConfPatternMerger();

		// Reset the merging
		virtual void		reset();

		// Accumulate some pattern (from the merging list?!)
		virtual void		add(const Pattern& pattern);

		// Copy the merged result to the given pattern
		virtual void		merge(Pattern& pattern) const;

	private:

		//////////////////////////////////////////////////////////////////
		// Attributes

		double			m_max_confidence;
		Pattern			m_pattern;
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

		// Add a pattern list
		void			add(const PatternList& lpatterns);

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
	//	- contains the list of candidate patterns, plus other representations
	//		(confidence map, usage map, hits map ...) automatically updated
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
		bool			reset(int image_w, int image_h, double model_threshold);
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
		const PatternList&      getPatternList() const { return m_patterns; }
		int**                   getConfidenceMap() const { return m_table_confidence; }
		unsigned char**         getUsageMap() const { return m_table_usage; }
		int**			getHitsMap() const { return m_table_hits; }
		int			getImageW() const { return m_image_w; }
		int			getImageH() const { return m_image_h; }

		/////////////////////////////////////////////////////////////////////////

	private:

		/////////////////////////////////////////////////////////////////////////

		// Normalize&Scale some model confidence
		int			normScaleConfidence(double confidence) const
		{
			return (int)(0.5 + (confidence - m_model_threshold) * 1000.0);
		}

		// Rescale&UnNormalize some model confidence
		double			unNormScaleConfidence(int ns_confidence) const
		{
			return m_model_threshold + ns_confidence * 0.001;
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
		double			m_model_threshold;

		// All patterns (also keeps track of the best patterns)
		PatternList		m_patterns;

		// Tables: normalized summed confidence, used sub-window corners&centers, hit counts
		int**			m_table_confidence;	// [m_image_w x m_image_h]
		unsigned char**		m_table_usage;	        // [m_image_w x m_image_h]
		int**			m_table_hits;
	};
}

#endif

