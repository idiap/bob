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

		// Check if it has the same sub-window as another pattern
		bool		isEqual(const Pattern& other) const
		{
			return 	m_x == other.m_x &&
				m_y == other.m_y &&
				m_w == other.m_w &&
				m_h == other.m_h;
		}

		// Returns the percentage of the overlapping area of intersection with another one
		int		getOverlap(const Pattern& other, bool ignoreInclusion = true) const;
		static int	getOverlap(	int x1, int y1, int w1, int h1,
						int x2, int y2, int w2, int h2,
						bool ignoreInclusion = true);

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
		PatternList();

		// Destructor
		~PatternList();

		// Add a new pattern - returns a reference to the stored pattern
		Pattern&		add(const Pattern& pattern, bool checkDuplicates = false);

		// Add a pattern list
		void			add(const PatternList& lpatterns, bool checkDuplicates = false);

		// Copy the data from a pattern list
		void			copy(const PatternList& lpatterns, bool checkDuplicates = false);

		// Invalidates all stored patterns (they are not deallocated, but ready to be used again)
		void			clear();

		// Deallocate all stored patterns
		void			deallocate();

		// Access functions
		bool			isEmpty() const;
		int			size() const;
		int			capacity() const;
		const Pattern&		get(int index) const;

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
}

#endif

