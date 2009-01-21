#include "MeanShiftSelector.h"

namespace Torch
{

/////////////////////////////////////////////////////////////////////////
// Constructor

MeanShiftSelector::MeanShiftSelector()
	:	m_LSH_L(0), m_LSH_hashTables(0)
{
}

/////////////////////////////////////////////////////////////////////////
// Destructor

MeanShiftSelector::~MeanShiftSelector()
{
	deallocate();
}

/////////////////////////////////////////////////////////////////////////
// Delete all stored patterns

void MeanShiftSelector::clear()
{
	Selector::clear();
}

/////////////////////////////////////////////////////////////////////////
// Deallocate the buffers

void MeanShiftSelector::deallocate()
{
	delete[] m_LSH_hashTables;
	m_LSH_hashTables = 0;
	m_LSH_L = 0;
}

/////////////////////////////////////////////////////////////////////////
// Constructor

MeanShiftSelector::LSH_HashTable::LSH_HashTable(int K)
	:	m_d(0),
		m_v(0),
		m_K(0)
{
	resize(K);
}

/////////////////////////////////////////////////////////////////////////
// Destructor

MeanShiftSelector::LSH_HashTable::~LSH_HashTable()
{
	delete[] m_d;
	delete[] m_v;
}

/////////////////////////////////////////////////////////////////////////
// Resize the number of inequalities

void MeanShiftSelector::LSH_HashTable::resize(int K)
{
	if (K != m_K && K > 0)
	{
		delete[] m_d;
		delete[] m_v;

		m_K = K;
		m_d = new unsigned char[m_K];
		m_v = new short[m_K];
	}
}

/////////////////////////////////////////////////////////////////////////
// Reset the hash table and generate inequalities

void MeanShiftSelector::LSH_HashTable::reset(int image_w, int image_h)
{
	for (int i = 0; i < HashSize; i ++)
	{
		m_values[i].m_size = 0;
	}

	srand((unsigned int)time(0));
	for (int k = 0; k < m_K; k ++)
	{
		m_d[k] = rand() % 4;	// cx, cy, w, h
		m_v[k] = (m_d[k] % 2 == 0) ?
				(rand() % image_w) // cx, w
				:
				(rand() % image_h);// cy, h
	}
}

/////////////////////////////////////////////////////////////////////////
// Add a new pattern

void MeanShiftSelector::LSH_HashTable::add(const Pattern& pattern, int index_pattern)
{
	const int cx = pattern.m_x + pattern.m_w / 2;
	const int cy = pattern.m_y + pattern.m_h / 2;

	// Generate the K-size boolean vector and hash it (DJB hashing)
	// See: http://www.partow.net/programming/hashfunctions/#DEKHashFunction
	unsigned int hash = 5381;
	for (int k = 0; k < m_K; k ++)
	{
		unsigned int cmp = 0;
		switch (m_d[k])
		{
		case 0:	// cx
			cmp = (cx <= m_v[k]) ? 1 : 0;
			break;

		case 1:	// cy
			cmp = (cy <= m_v[k]) ? 1 : 0;
			break;

		case 2:	// w
			cmp = pattern.m_w <= m_v[k];
			break;

		case 3:	// h
		default:
			cmp = pattern.m_h <= m_v[k];
			break;
		}

		hash = ((hash << 5) + hash) + (cmp + 1);
	}
	hash = hash % HashSize;

	// Add the pattern to the hashed cell
	Value& value = m_values[hash];
	if (value.m_size >= value.m_capacity)
	{
		value.resize();
	}
	value.m_indexes[value.m_size ++] = index_pattern;
}

/////////////////////////////////////////////////////////////////////////
// Constructor

MeanShiftSelector::LSH_HashTable::Value::Value()
	:	m_indexes(0),
		m_size(0), m_capacity(0)
{
}

/////////////////////////////////////////////////////////////////////////
// Destructor

MeanShiftSelector::LSH_HashTable::Value::~Value()
{
	delete[] m_indexes;
	m_indexes = 0;
}

/////////////////////////////////////////////////////////////////////////
// Resize the indexes already stored to accomodate new ones

void MeanShiftSelector::LSH_HashTable::Value::resize()
{
	const int delta = m_size * 8 / 5;

	int* new_indexes = new int[m_size + delta];
	for (int i = 0; i < m_size; i ++)
	{
		new_indexes[i] = m_indexes[i];
	}
	delete[] m_indexes;
	m_indexes = new_indexes;

	m_capacity += delta;
}

/////////////////////////////////////////////////////////////////////////
// Initialize the LSH structures

void MeanShiftSelector::initLSH(const PatternList& lpatterns, int image_w, int image_h)
{
	// TODO: estimate K & L!!!
	static const int K = 20;
	static const int L = 4;

	// Allocate and reset the LSH hash tables
	if (L != m_LSH_L)
	{
		deallocate();

		m_LSH_L = L;
		m_LSH_hashTables = new LSH_HashTable[m_LSH_L];
	}

	for (int l = 0; l < m_LSH_L; l ++)
	{
		LSH_HashTable& table = m_LSH_hashTables[l];
		table.resize(K);
		table.reset(image_w, image_h);
	}

	// Compute the hash tables
	const int n_patterns = lpatterns.size();
	for (int i = 0; i < n_patterns; i ++)
	{
		const Pattern& pattern = lpatterns.get(i);
		for (int l = 0; l < m_LSH_L; l ++)
		{
			m_LSH_hashTables[l].add(pattern, i);
		}
	}
}

/////////////////////////////////////////////////////////////////////////
// Get the closest points to the given one (using LSH tables)
// Returns the number of found points (less or equal than <max_size>)

int MeanShiftSelector::getClosest(const Pattern& pattern, int* iclosest, int* iclosestBuf, int max_size)
{
	// TODO: clear the result array (-1)
	// TODO: do the intersection over the L partitions

	return 0;
}

/////////////////////////////////////////////////////////////////////////
// Process the list of candidate sub-windows and select the best ones
// (this will accumulate them to the pattern list)

bool MeanShiftSelector::process(const PatternSpace& candidates)
{
	// Get parameters
	const bool verbose = getBOption("verbose");
	const int max_n_iters = 100;	// Maximum number of iterations
	const int kclosest = 8;		// Used for dynamic bandwidth computation (see paper)

	// Check parameters
	if (candidates.isEmpty() == true)
	{
		if (verbose == true)
		{
			Torch::message("MeanShiftSelector::process - the pattern space is empty!\n");
		}
		return true;
	}

	/*
	////////////////////////////////////////////////////////////////
	// Cluster the patterns using AMS (Adaptive Bandwidth Mean Shift)
	//	and Locality Sensitive Hashing (LSH)

	PatternList lpatterns;
	const PatternList& candidates_list = candidates.getPatternList();

	// LSH (initialization)
	initLSH(candidates_list, candidates.getImageW(), candidates.getImageH());

	// Buffers to copy the indexes of the closest points for some partition
	static const int MaxNoClosestPoints = 64;
	int iclosest[MaxNoClosestPoints];
	int iclosestBuf[MaxNoClosestPoints];

	// AMS (for each point until convergence)
	const int n_patterns = candidates_list.size();
	for (int i = 0; i < n_patterns; i ++)
	{
		const Pattern& crt_pattern = candidates_list.get(i);

		int crt_cx = crt_pattern.getCenterX();
		int crt_cy = crt_pattern.getCenterY();
		int crt_w = crt_pattern.m_w;
		int crt_h = crt_pattern.m_h;

		// Compute the bandwidth for this sample (from the center till the k-closest)
		const int n_points = getClosest(crt_pattern, iclosest, iclosestBuf, MaxNoClosestPoints);
		const Pattern& kclosest_pattern =
			candidates_list.get(iclosest[n_points < kclosest ? (n_points - 1) : (kclosest - 1)]);

		const int rad_x = abs(crt_cx - kclosest_pattern.getCenterX());
		const int rad_y = abs(crt_cy - kclosest_pattern.getCenterY());

		// Iterate until convergence (or the maximum number of iterations was reached)
		int n_iters = 0;
		while ((n_iters ++) < max_n_iters)
		{
			int sum_cx = 0;
			int sum_cy = 0;
			int sum_w = 0;
			int sum_h = 0;
			int sum_cnt = 0;

			// Compute the mean shift looking for neighbour patterns
			// TODO: use LSH!
			// TODO: allocate a buffer where the closest points to be returned!
			for (int j = 0; j < n_patterns; j ++)
			{
				const Pattern& pattern = candidates_list.get(j);

				const int cx = pattern.getCenterX();
				const int cy = pattern.getCenterY();
				if (	abs(crt_cx - cx) < rad_x &&
					abs(crt_cy - cy) < rad_y)
				{
					sum_cx += cx;
					sum_cy += cy;
					sum_w += pattern.m_w;
					sum_h += pattern.m_h;
					sum_cnt ++;
				}
			}

			// Update the current position (if not converged yet)
			if (sum_cnt <= 1)
			{
				break;
			}

			const double inv = 1.0 / sum_cnt;
			const int new_crt_cx = FixI(inv * sum_cx);
			const int new_crt_cy = FixI(inv * sum_cy);
			const int new_crt_w = FixI(inv * sum_w);
			const int new_crt_h = FixI(inv * sum_h);

			const int diff_cx = new_crt_cx - crt_cx;
			const int diff_cy = new_crt_cy - crt_cy;
			const int diff_w = new_crt_w - crt_w;
			const int diff_h = new_crt_h - crt_h;

			if (	diff_cx * diff_cx <= 1 &&
				diff_cy * diff_cy <= 1 &&
				diff_w * diff_w <= 1 &&
				diff_h * diff_h <= 1)
			{
				break;
			}

			crt_cx = new_crt_cx;
			crt_cy = new_crt_cy;
			crt_w = new_crt_w;
			crt_h = new_crt_h;
		}

		// Add the converged point (~ density mode) to the list
		lpatterns.add(Pattern(crt_cx - crt_w / 2, crt_cy - crt_h / 2, crt_w, crt_h, crt_pattern.m_confidence));
	}

	// TODO: need a way to decide the clusters!
	// TODO: without checking if some SWs overlap with O(N^2) complexity

	// Accumulate the detections to the result
	m_patterns.add(lpatterns);
	*/

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////

}

