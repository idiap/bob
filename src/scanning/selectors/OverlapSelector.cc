#include "OverlapSelector.h"

namespace Torch
{

/////////////////////////////////////////////////////////////////////////
// Constructor

OverlapSelector::OverlapSelector(PatternMerger* merger)
	:	m_overlappingTable(0),
		m_patternBuffer(0),
		m_patternTags(0),
		m_n_allocated_patterns(0),
		m_surf_threshold(60),
		m_merger(merger)
{
	addBOption("iterative", false, "merge the sub-windows in one pass or iteratively");
}

/////////////////////////////////////////////////////////////////////////
// Destructor

OverlapSelector::~OverlapSelector()
{
	deallocate();
}

/////////////////////////////////////////////////////////////////////////
// Deallocate buffers

void OverlapSelector::deallocate()
{
	delete[] m_overlappingTable;
	m_overlappingTable = 0;
	delete[] m_patternBuffer;
	m_patternBuffer = 0;
	delete[] m_patternTags;
	m_patternTags = 0;
	m_n_allocated_patterns = 0;
}

/////////////////////////////////////////////////////////////////////////
// Delete all stored patterns

void OverlapSelector::clear()
{
	Selector::clear();
}

/////////////////////////////////////////////////////////////////////////
// Change the merging strategy

bool OverlapSelector::setMerger(PatternMerger* merger)
{
	if (merger == 0)
	{
		Torch::message("OverlapSelector::setMerger - invalid merger!\n");
		return false;
	}

	// OK
	m_merger = merger;
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Resize the buffers to accomodate new candidates

void OverlapSelector::resize(int new_size)
{
	if (new_size > m_n_allocated_patterns)
	{
		deallocate();

		m_overlappingTable = new OverlapInfo[new_size];
		m_patternBuffer = new Pattern[new_size];
		m_patternTags = new unsigned char[new_size];
		m_n_allocated_patterns = new_size;
	}
	m_n_patterns = new_size;
}

/////////////////////////////////////////////////////////////////////////
// Initialize buffers from some pattern list

int sortConfDesc(const void* sw1, const void* sw2)
{
	const Pattern* patt1 = (Pattern*)sw1;
	const Pattern* patt2 = (Pattern*)sw2;

	return patt1->m_confidence < patt2->m_confidence ? 1 :
		(patt1->m_confidence > patt2->m_confidence ? -1 : 0);
}

void OverlapSelector::init(const PatternList& patterns)
{
	const int n_patterns = patterns.size();
	resize(n_patterns);

	// Create a copy of the patterns and sort them descending over the model confidence
	for (int i = 0; i < n_patterns; i ++)
	{
		m_patternBuffer[i].copy(patterns.get(i));
		m_patternTags[i] = false;
	}
	qsort(m_patternBuffer, n_patterns, sizeof(Pattern), sortConfDesc);

	// Compute the overalapping table
	for (int i = 0; i < n_patterns; i ++)
	{
		const Pattern& source = m_patternBuffer[i];
		OverlapInfo& info = m_overlappingTable[i];

		info.reset(m_surf_threshold);
		for (int j = 0; j < n_patterns; j ++)
			if (i != j)
		{
			info.add(source, m_patternBuffer[j], j);
		}
	}
}

/////////////////////////////////////////////////////////////////////////
// Add a new pattern, if it overlaps enough

void OverlapSelector::OverlapInfo::add(const Pattern& source, const Pattern& test_patt, int index_patt)
{
	const int surf = source.getOverlap(test_patt);
	if (surf >= m_surf_threshold)
	{
		// Keep the overlapping pattern list ordered descending over the overlapping percentage
		const int insert_index = find(surf, 0, m_n_overlaps);
		if (insert_index >= m_n_overlaps)
		{
			if (m_n_overlaps < MaxNoOverlaps)
			{
				// Insert at the end (no shift is needed)
				m_overlaps[insert_index].m_index = index_patt;
				m_overlaps[insert_index].m_surf = surf;
				m_n_overlaps ++;
			}
		}
		else
		{
			if (m_n_overlaps < MaxNoOverlaps)
			{
				m_n_overlaps ++;
			}

			// Insert in the middle (shift to the right first)
			for (int j = m_n_overlaps - 1; j > insert_index; j --)
			{
				m_overlaps[j] = m_overlaps[j - 1];
			}
			m_overlaps[insert_index].m_index = index_patt;
			m_overlaps[insert_index].m_surf = surf;
		}
	}
}

/////////////////////////////////////////////////////////////////////////
// Returns the index where to insert a new pattern given its surface overlapping
// (as to have them ordered descending over the ovelapping surface)

int OverlapSelector::OverlapInfo::find(int surf, int index_start, int index_stop) const
{
	if (index_start == index_stop)
	{
		return index_start;
	}
	else if (index_start + 1 == index_stop)
	{
		return m_overlaps[index_start].m_surf >= surf ?
			index_stop : index_start;
	}
	else
	{
		const int index_middle = (index_start + index_stop) / 2;
		return m_overlaps[index_middle].m_surf >= surf ?
			find(surf, index_middle + 1, index_stop) :
			find(surf, index_start, index_middle);
	}
}

/////////////////////////////////////////////////////////////////////////
// Print the overlapping table

void OverlapSelector::printOverlapTable() const
{
	print("------ OVERLAPPING TABLE --------------\n");
	for (int i = 0; i < m_n_patterns; i ++)
	{
		const OverlapInfo& info = m_overlappingTable[i];

		print("[%d/%d - %d overlaps]: ", i + 1, m_n_patterns, info.m_n_overlaps);
		for (int j = 0; j < info.m_n_overlaps; j ++)
		{
			print("%d:%d, ", info.m_overlaps[j].m_index + 1, info.m_overlaps[j].m_surf);
		}
		print("\n");
	}
	print("---------------------------------------\n");
}

/////////////////////////////////////////////////////////////////////////
// Process the list of candidate sub-windows and select the best ones
// (this will accumulate them to the pattern list)

bool OverlapSelector::process(const PatternSpace& candidates)
{
	// Check parameters
	if (m_merger == 0)
	{
		Torch::message("OverlapSelector::process - no merger specified!\n");
		return false;
	}
	if (candidates.isEmpty() == true)
	{
		Torch::message("OverlapSelector::process - the pattern space is empty!\n");
		return true;	//
	}

	const bool verbose = getBOption("verbose");
	const bool iterative = getBOption("iterative");

	Pattern pattern;

	// Initialize the buffers (sort a copy descending, build the overlapping table ...)
	init(candidates.getPatternList());
	if (verbose == true)
	{
		printOverlapTable();
	}

	// Merge detections ... until all the candidates are covered
	int n_steps = 0;
	bool stop = false;
	while (stop == false)
	{
		n_steps ++;
		if (verbose == true)
		{
			print("OverlapSelector: step no. [%d] ...\n", n_steps);
		}

		// Delete the old merged patterns (if any), it's used also as a temporary list!
		m_patterns.clear();

		// Merge the overlapping detections
		int n_merged_patterns = 0;
		for (int i = 0; i < m_n_patterns; i ++)
			if (m_patternTags[i] == false)
		{
			// Tag the detection, it should not be considered one more time!
			m_patternTags[i] = true;
			n_merged_patterns ++;

			OverlapInfo& info = m_overlappingTable[i];
			if (info.m_n_overlaps == 0)
			{
				// Nothing to merge!
				m_patterns.add(m_patternBuffer[i]);
			}
			else
			{
				const int n_overlaps_to_merge =
					iterative == true ?
						1  	//merge only the sub-windows that overlap most
						:
						info.m_n_overlaps;	//merge all the overlapping detections

				// Merge the overlapping detections
				m_merger->reset();
				m_merger->add(m_patternBuffer[i]);
				for (int j = 0; j < n_overlaps_to_merge; j ++)
				{
					const int k = info.m_overlaps[j].m_index;
					if (m_patternTags[k] == false)
					{
						m_merger->add(m_patternBuffer[k]);
						m_patternTags[k] = true;
						n_merged_patterns ++;
					}
				}
				m_merger->merge(pattern);

				m_patterns.add(pattern);
			}
		}

		if (verbose == true)
		{
			print("\t[%d/%d] merged patterns\n", n_merged_patterns, m_n_patterns);
		}

		// Re-initialize the buffers (as above, but now from the temporary list)
		init(m_patterns);
		if (verbose == true)
		{
			printOverlapTable();
		}

		// Check if more patterns must be merged
		stop = true;
		for (int i = 0; i < m_n_patterns; i ++)
			if (m_overlappingTable[i].m_n_overlaps > 0)
			{
				stop = false;
				break;
			}
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////

}

