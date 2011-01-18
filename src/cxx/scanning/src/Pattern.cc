#include "scanning/Pattern.h"
#include "core/general.h"

namespace Torch
{

/////////////////////////////////////////////////////////////////////////
// Returns the percentage of the overlapping area of intersection with another one

int Pattern::getOverlap(const Pattern& other, bool ignoreInclusion) const
{
	return getOverlap(m_x, m_y, m_w, m_h, other.m_x, other.m_y, other.m_w, other.m_h, ignoreInclusion);
}

int Pattern::getOverlap(	int x1, int y1, int w1, int h1,
				int x2, int y2, int w2, int h2,
				bool ignoreInclusion)
{
	int x_min, y_min;
	int x_max, y_max;
	int overlap_x_min, overlap_y_min;
	int overlap_x_max, overlap_y_max;

	// Check the corners - left
	if (x1 <= x2)
	{
		x_min = x1;
		overlap_x_min = x2;
	}
	else
	{
		x_min = x2;
		overlap_x_min = x1;
	}

	// Check the corners - top
	if (y1 <= y2)
	{
		y_min = y1;
		overlap_y_min = y2;
	}
	else
	{
		y_min = y2;
		overlap_y_min = y1;
	}

	// Check the corners - right
	if (x1 + w1 >= x2 + w2)
	{
		x_max = x1 + w1;
		overlap_x_max = x2 + w2;
	}
	else
	{
		x_max = x2 + w2;
		overlap_x_max = x1 + w1;
	}

	// Check the corners - bottom
	if (y1 + h1 >= y2 + h2)
	{
		y_max = y1 + h1;
		overlap_y_max = y2 + h2;
	}
	else
	{
		y_max = y2 + h2;
		overlap_y_max = y1 + h1;
	}

	// No intersection
	if (overlap_x_max < overlap_x_min || overlap_y_max < overlap_y_min)
	{
		return 0;
	}
	else if (x_max - x_min > w1 + w2 || y_max - y_min > h1 + h2)
	{
	   	return 0;
	}

	// Inclusion
	else if (	ignoreInclusion == false &&
			((x_max - x_min == w1 && y_max - y_min == h1) ||
			(x_max - x_min == w2 && y_max - y_min == h2)))
	{
		return 100;
	}

	// Some intersection
	else
	{
		if (h2 * w2 > h1 * w1)
		{
			return 100 * (overlap_x_max - overlap_x_min) * (overlap_y_max - overlap_y_min) / (h2 * w2);
		}
	   	else
	   	{
			return 100 * (overlap_x_max - overlap_x_min) * (overlap_y_max - overlap_y_min) / (h1 * w1);
	    	}
	}
}

/////////////////////////////////////////////////////////////////////////
// Constructor

AveragePatternMerger::AveragePatternMerger()
	:	m_sum_cx(0.0),
		m_sum_cy(0.0),
		m_sum_w(0.0),
		m_sum_h(0.0),
		m_sum_confidence(0.0),
		m_count(0)
{
}

/////////////////////////////////////////////////////////////////////////
// Destructor

AveragePatternMerger::~AveragePatternMerger()
{
}

/////////////////////////////////////////////////////////////////////////
// Reset the merging

void AveragePatternMerger::reset()
{
	m_sum_cx = 0.0;
	m_sum_cy = 0.0;
	m_sum_w = 0.0;
	m_sum_h = 0.0;
	m_sum_confidence = 0.0;
	m_count = 0;
}

/////////////////////////////////////////////////////////////////////////
// Accumulate some pattern (from the merging list?!)

void AveragePatternMerger::add(const Pattern& pattern)
{
	m_sum_cx += pattern.m_x + 0.5 * pattern.m_w;
	m_sum_cy += pattern.m_y + 0.5 * pattern.m_h;
	m_sum_w += pattern.m_w;
	m_sum_h += pattern.m_h;
	m_sum_confidence += pattern.m_confidence;
	m_count ++;
}

/////////////////////////////////////////////////////////////////////////
// Copy the merged result to the given pattern

void AveragePatternMerger::merge(Pattern& pattern) const
{
	if (m_count > 0)
	{
		const double inv = 1.0 / m_count;
		pattern.m_x = FixI(inv * (m_sum_cx - m_sum_w * 0.5));
		pattern.m_y = FixI(inv * (m_sum_cy - m_sum_h * 0.5));
		pattern.m_w = FixI(inv * m_sum_w);
		pattern.m_h = FixI(inv * m_sum_h);
		pattern.m_confidence = inv * m_sum_confidence;
		pattern.m_activation = m_count;
	}
}

/////////////////////////////////////////////////////////////////////////
// Constructor

ConfWeightedPatternMerger::ConfWeightedPatternMerger()
	:	m_sum_cx(0.0),
		m_sum_cy(0.0),
		m_sum_w(0.0),
		m_sum_h(0.0),
		m_sum_confidence(0.0),
		m_count(0)
{
}

/////////////////////////////////////////////////////////////////////////
// Destructor

ConfWeightedPatternMerger::~ConfWeightedPatternMerger()
{
}

/////////////////////////////////////////////////////////////////////////
// Reset the merging

void ConfWeightedPatternMerger::reset()
{
	m_sum_cx = 0.0;
	m_sum_cy = 0.0;
	m_sum_w = 0.0;
	m_sum_h = 0.0;
	m_sum_confidence = 0.0;
	m_count = 0;
}

/////////////////////////////////////////////////////////////////////////
// Accumulate some pattern (from the merging list?!)

void ConfWeightedPatternMerger::add(const Pattern& pattern)
{
	// Try to avoid the cases where the confidence is 0.0
	static const double delta_confidence = 10000.0;
	const double confidence = delta_confidence + pattern.m_confidence;

	m_sum_cx += confidence * (pattern.m_x + 0.5 * pattern.m_w);
	m_sum_cy += confidence * (pattern.m_y + 0.5 * pattern.m_h);
	m_sum_w += confidence * pattern.m_w;
	m_sum_h += confidence * pattern.m_h;
	m_sum_confidence += confidence;
	m_count ++;
}

/////////////////////////////////////////////////////////////////////////
// Copy the merged result to the given pattern

void ConfWeightedPatternMerger::merge(Pattern& pattern) const
{
	if (m_count > 0)
	{
		static const double delta_confidence = 10000.0;

		const double inv = 1.0 / m_sum_confidence;
		pattern.m_x = FixI(inv * (m_sum_cx - m_sum_w * 0.5));
		pattern.m_y = FixI(inv * (m_sum_cy - m_sum_h * 0.5));
		pattern.m_w = FixI(inv * m_sum_w);
		pattern.m_h = FixI(inv * m_sum_h);
		pattern.m_confidence = m_sum_confidence / m_count - delta_confidence;
		pattern.m_activation = m_count;
	}
}

/////////////////////////////////////////////////////////////////////////
// Constructor

MaxConfPatternMerger::MaxConfPatternMerger()
	:	m_max_confidence(-100000.0)
{
}

/////////////////////////////////////////////////////////////////////////
// Destructor

MaxConfPatternMerger::~MaxConfPatternMerger()
{
}

/////////////////////////////////////////////////////////////////////////
// Reset the merging

void MaxConfPatternMerger::reset()
{
	m_max_confidence = -100000.0;
	m_pattern.m_activation = 0;
}

/////////////////////////////////////////////////////////////////////////
// Accumulate some pattern (from the merging list?!)

void MaxConfPatternMerger::add(const Pattern& pattern)
{
	if (pattern.m_confidence > m_max_confidence)
	{
		m_max_confidence = pattern.m_confidence;
		const short last_activation = m_pattern.m_activation;
		m_pattern.copy(pattern);
		m_pattern.m_activation = last_activation + 1;
	}
}

/////////////////////////////////////////////////////////////////////////
// Copy the merged result to the given pattern

void MaxConfPatternMerger::merge(Pattern& pattern) const
{
	pattern.copy(m_pattern);
}

/////////////////////////////////////////////////////////////////////////
// Constructor

PatternList::PatternList()
	:	m_nodes(new Node*[NodeArraySizeIncrement]),
		m_idx_last_node(0),
		m_n_allocated_nodes(NodeArraySizeIncrement),
		m_n_used_patterns(0),
		m_n_allocated_patterns(0)
{
	// Allocate the pointers to nodes
	for (int i = 0; i < NodeArraySizeIncrement; i ++)
	{
		m_nodes[i] = 0;
	}
}

/////////////////////////////////////////////////////////////////////////
// Destructor

PatternList::~PatternList()
{
	deallocate();
	delete[] m_nodes;
}

/////////////////////////////////////////////////////////////////////////
// Add a new pattern - returns a reference to the stored pattern

Pattern& PatternList::add(const Pattern& pattern, bool checkDuplicates)
{
	// Check for duplicates if required
	if (checkDuplicates == true && isEmpty() == false)
	{
		for (int i = 0; i <= m_idx_last_node; i ++)
		{
			Node& node = *m_nodes[i];
			for (int j = 0; j < node.m_n_used; j ++)
			{
				Pattern& cmp_pattern = node.m_patterns[j];
				if (cmp_pattern.isEqual(pattern) == true)
				{
					return cmp_pattern;
				}
			}
		}
	}

	Node* add_node = 0;

	// Find the last node that has some available space
	while (m_idx_last_node < m_n_allocated_nodes)
	{
		Node** last_node = &m_nodes[m_idx_last_node];

		// Check if there is some free space in the last node
		if (*last_node == 0)
		{
			*last_node = new Node;
			m_n_allocated_patterns += PatternArraySize;
			add_node = *last_node;
			break;
		}
		else if ((*last_node)->m_n_used < (*last_node)->m_n_allocated)
		{
			add_node = *last_node;
			break;
		}

		// Try with the next node
		m_idx_last_node ++;
	}

	// Need to allocate new nodes
	if (m_idx_last_node == m_n_allocated_nodes)
	{
		// Make a copy of the current pointers to nodes
		Node** temp = new Node*[m_n_allocated_nodes + NodeArraySizeIncrement];
		for (int i = 0; i < m_n_allocated_nodes; i ++)
		{
			temp[i] = m_nodes[i];
		}
		for (int i = 0; i < NodeArraySizeIncrement; i ++)
		{
			temp[m_n_allocated_nodes + i] = 0;
		}

		// Make the copy the current pointers
		delete[] m_nodes;
		m_nodes = temp;
		m_n_allocated_nodes += NodeArraySizeIncrement;

		// Allocate the new node
		m_nodes[m_idx_last_node] = new Node;
		m_n_allocated_patterns += PatternArraySize;
		add_node = m_nodes[m_idx_last_node];
	}

	// Add it to the last node (it has enough allocated memory)
	m_n_used_patterns ++;
	return add_node->add(pattern);
}

/////////////////////////////////////////////////////////////////////////
// Add a pattern list

void PatternList::add(const PatternList& lpatterns, bool checkDuplicates)
{
	const int n_patterns = lpatterns.size();
	for (int i = 0; i < n_patterns; i ++)
	{
		add(lpatterns.get(i), checkDuplicates);
	}
}

/////////////////////////////////////////////////////////////////////////
// Copy the data from a pattern list

void PatternList::copy(const PatternList& lpatterns, bool checkDuplicates)
{
	clear();
	add(lpatterns, checkDuplicates);
}

/////////////////////////////////////////////////////////////////////////
// Invalidates all stored patterns (they are not deallocated, but ready to be used again)

void PatternList::clear()
{
	for (int i = 0; i < m_n_allocated_nodes; i ++)
	{
		if (m_nodes[i] != 0)
			m_nodes[i]->m_n_used = 0;
		else
			break;
	}

	m_n_used_patterns = 0;
	m_idx_last_node = 0;
}

/////////////////////////////////////////////////////////////////////////
// Deallocate all stored patterns

void PatternList::deallocate()
{
	// Delete allocated nodes
	for (int i = 0; i < m_n_allocated_nodes; i ++)
	{
		delete m_nodes[i];
		m_nodes[i] = 0;
	}

	// Reset statistics
	m_n_used_patterns = 0;
	m_n_allocated_patterns = 0;
	m_idx_last_node = 0;
}

/////////////////////////////////////////////////////////////////////////
// Access functions

bool PatternList::isEmpty() const
{
	return m_n_used_patterns == 0;
}

int PatternList::size() const
{
	return m_n_used_patterns;
}

int PatternList::capacity() const
{
	return m_n_allocated_patterns;
}

const Pattern& PatternList::get(int index) const
{
	return m_nodes[getNodeIndex(index)]->m_patterns[getPatternIndex(index)];
}

/////////////////////////////////////////////////////////////////////////
// Constructor

PatternList::Node::Node()
	:	m_patterns(new Pattern[PatternArraySize]),
		m_n_used(0),
		m_n_allocated(PatternArraySize)
{
}

/////////////////////////////////////////////////////////////////////////
// Destructor

PatternList::Node::~Node()
{
	delete[] m_patterns;
}

/////////////////////////////////////////////////////////////////////////
// Add a new pattern - returns a pattern
// (presuming that there is enough allocated memory)

Pattern& PatternList::Node::add(const Pattern& pattern)
{
	m_patterns[m_n_used ++].copy(pattern);
	return m_patterns[m_n_used - 1];
}

/////////////////////////////////////////////////////////////////////////
}
