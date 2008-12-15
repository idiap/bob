#include "Pattern.h"
#include "general.h"

namespace Torch
{

/////////////////////////////////////////////////////////////////////////
// Constructor

PatternList::PatternList(bool trackBestPatterns, int maxBestPatterns)
	:	m_nodes(new Node*[NodeArraySizeIncrement]),
		m_idx_last_node(0),
		m_n_allocated_nodes(NodeArraySizeIncrement),

		m_best_activated(false),
		m_best_indexes(0),
		m_best_size(0),
		m_best_max_size(0),

		m_n_used_patterns(0),
		m_n_allocated_patterns(0)
{
	// Allocate the pointers to nodes
	for (int i = 0; i < NodeArraySizeIncrement; i ++)
	{
		m_nodes[i] = 0;
	}

	// Allocate the best patterns
	resetBestPatterns(trackBestPatterns, maxBestPatterns);
}

/////////////////////////////////////////////////////////////////////////
// Destructor

PatternList::~PatternList()
{
	deallocate();
	delete[] m_nodes;
}

/////////////////////////////////////////////////////////////////////////
// Add a new pattern

Pattern& PatternList::add(const Pattern& pattern)
{
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

	// If the tracking of the best points is activated,
	//	need to rearrange the list of the best patterns
	if (m_best_activated == true)
	{
		addBest(pattern, m_idx_last_node * PatternArraySize + add_node->m_n_used);
	}

	// Add it to the last node (it has enough allocated memory)
	m_n_used_patterns ++;
	return add_node->add(pattern);
}

/////////////////////////////////////////////////////////////////////////
// Add a new pattern in the ordered list of the best patterns

void PatternList::addBest(const Pattern& pattern, int index_pattern)
{
	// Find the position where it should be inserted
	const int insert_pos = m_best_size == 0 ? 0 : findBest(pattern, 0, m_best_size - 1);

	if (insert_pos >= m_best_size)
	{
		// Should be added at the end, only if enough space
		if (m_best_size < m_best_max_size)
		{
			m_best_indexes[m_best_size ++] = index_pattern;
		}
	}

	else
	{
		// Should be inserted at <index> position, need to shift the ones after
		if (m_best_size < m_best_max_size)
		{
			m_best_size ++;
		}
		for (int i = m_best_size; i > insert_pos; i --)
		{
			m_best_indexes[i] = m_best_indexes[i - 1];
		}
		m_best_indexes[insert_pos] = index_pattern;
	}
}

/////////////////////////////////////////////////////////////////////////
// Finds the position where some pattern should be inserted in order to keep the list of
//	the best ones as sorted

int PatternList::findBest(const Pattern& pattern, int index1, int index2) const
{
	//////////////////////////////////////////////////////////
	// Binary search, descending order
	//////////////////////////////////////////////////////////

	if (index1 >= index2)
	{
		return index1;
	}

	const int indexm = (index1 + index2) / 2;

	if (get(m_best_indexes[indexm]).m_confidence >= pattern.m_confidence)
	{
		return findBest(pattern, indexm + 1, index2);
	}
	else
	{
		return findBest(pattern, index1, indexm);
	}
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

	for (int i = 0; i < m_best_max_size; i ++)
	{
		m_best_indexes[i] = -1;
	}

	m_n_used_patterns = 0;
	m_idx_last_node = 0;
	m_best_size = 0;
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

	// Delete best patterns
	delete[] m_best_indexes;
	m_best_indexes = 0;
	m_best_size = 0;

	// Reset statistics
	m_n_used_patterns = 0;
	m_n_allocated_patterns = 0;
	m_idx_last_node = 0;
}

/////////////////////////////////////////////////////////////////////////
// Change if the best patterns are stored and their maximum number

bool PatternList::resetBestPatterns(bool trackBestPatterns, int maxBestPatterns)
{
	if (maxBestPatterns < 1)
	{
		return false;
	}

	// Need resizing ?!
	if (maxBestPatterns != m_best_max_size)
	{
		// Allocate the new indexes and copy the old ones
		int* temp = new int[maxBestPatterns + 1];
		for (int i = 0; i < maxBestPatterns; i ++)
		{
			if (i < m_best_size)
				temp[i] = m_best_indexes[i];
			else
				temp[i] = -1;
		}

		// Delete the old indexes and point to the new indexes
		delete[] m_best_indexes;
		m_best_indexes = temp;

		m_best_size = min(m_best_size, maxBestPatterns);
		m_best_max_size = maxBestPatterns;
	}

	// OK
	m_best_activated = trackBestPatterns;
	return true;
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

int PatternList::getNoBest() const
{
	return m_best_size;
}

int PatternList::getMaxNoBest() const
{
	return m_best_max_size;
}

const Pattern& PatternList::getBest(int index) const
{
	return get(m_best_indexes[index]);
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
// Constructor

PatternSpace::PatternSpace()
	:	m_image_w(0), m_image_h(0), m_model_threshold(0.0f),
		m_patterns(true),	// <true> for keeping track of the best patterns
		m_table_confidence(0), m_table_usage(0)
{
}

/////////////////////////////////////////////////////////////////////////
// Destructor

PatternSpace::~PatternSpace()
{
	deallocateTables();
	deallocatePatterns();
}

/////////////////////////////////////////////////////////////////////////
// Deallocate the allocated tables and pattern list

void PatternSpace::deallocateTables()
{
	for (int i = 0; i < m_image_w; i ++)
	{
		delete[] m_table_confidence[i];
		delete[] m_table_usage[i];
	}
	delete[] m_table_confidence;
	delete[] m_table_usage;
	m_table_confidence = 0;
	m_table_usage = 0;
}

void PatternSpace::deallocatePatterns()
{
	m_patterns.deallocate();
}

/////////////////////////////////////////////////////////////////////////
// Delete all stored patterns (but it's not deallocating the memory)

void PatternSpace::clear()
{
	// Clear the tables
	for (int i = 0; i < m_image_w; i ++)
		for (int j = 0; j < m_image_h; j ++)
		{
			m_table_confidence[i][j] = 0;
			m_table_usage[i][j] = 0;
		}

	// Delete patterns
	m_patterns.clear();
}

/////////////////////////////////////////////////////////////////////////
// Reset to a new context:
//	- images of different sizes
//	- new model threshold
//	- number of best points to keep track of

bool PatternSpace::reset(int image_w, int image_h, double model_threshold)
{
	// Check if the tables should be resized
	if (image_w != m_image_w || image_h != m_image_h)
	{
		if (image_w > 0 && image_h > 0)
		{
			// Dealocate the old tables
			deallocateTables();

			// Resize the tables
			m_image_w = image_w;
			m_image_h = image_h;

			m_table_confidence = new int*[m_image_w];
			m_table_usage = new unsigned char*[m_image_w];
			for (int i = 0; i < m_image_w; i ++)
			{
				m_table_confidence[i] = new int[m_image_h];
				m_table_usage[i] = new unsigned char[m_image_h];
			}
		}
		else
		{
			return false;
		}
	}

	// OK
	m_model_threshold = model_threshold;
	clear();
	return true;
}

bool PatternSpace::reset(int n_best)
{
	return m_patterns.resetBestPatterns(true, n_best);
}

/////////////////////////////////////////////////////////////////////////
// Add a new candidate pattern (sub-window coordinates + model confidence)

void PatternSpace::add(const Pattern& pattern)
{
	const int sw_x = pattern.m_x;
	const int sw_y = pattern.m_y;
	const int sw_w = pattern.m_w;
	const int sw_h = pattern.m_h;

	// Update the (normalized&scaled) confidence table
	const int ns_confidence = normScaleConfidence(pattern.m_confidence);
	for (int i = 0; i < sw_w; i ++)
		for (int j = 0; j < sw_h; j ++)
		{
			m_table_confidence[sw_x + i][sw_y + j] += ns_confidence;
		}

	// Update the usage table
	m_table_usage[sw_x][sw_y] = 0x01;
        m_table_usage[sw_x + sw_w][sw_y] = 0x01;
        m_table_usage[sw_x + sw_w][sw_y + sw_h] = 0x01;
        m_table_usage[sw_x][sw_y + sw_h] = 0x01;

        // One more pattern is stored
	m_patterns.add(pattern);
}

/////////////////////////////////////////////////////////////////////////
// Check if some scanning point is already stored

bool PatternSpace::hasPoint(int sw_x, int sw_y, int sw_w, int sw_h)
{
        // Check each detection (reliable, but slow)
        for (int i = 0; i < m_patterns.size(); i ++)
        {
                const Pattern& pattern = m_patterns.get(i);
                if (    pattern.m_x == sw_x ||
                        pattern.m_y == sw_y ||
                        pattern.m_w == sw_w ||
                        pattern.m_h == sw_h)
                {
                        return true;
                }
        }
        return false;

        // Not reliable, but fast!
        return  m_table_usage[sw_x][sw_y] == 0x01 &&
                m_table_usage[sw_x + sw_w][sw_y + sw_h] == 0x01 &&
                m_table_confidence[sw_x][sw_y] == m_table_confidence[sw_x + sw_w][sw_y + sw_h];
}

/////////////////////////////////////////////////////////////////////////

}
