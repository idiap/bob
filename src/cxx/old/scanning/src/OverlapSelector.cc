/**
 * @file cxx/old/scanning/src/OverlapSelector.cc
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
#include "scanning/OverlapSelector.h"

namespace Torch
{

/////////////////////////////////////////////////////////////////////////
// Constructor

OverlapSelector::OverlapSelector(PatternMerger* merger)
	:	m_overlappingTable(0),
		m_patternBuffer(0),
		m_patternTags(0),
		m_n_allocated_patterns(0),
		m_merger(merger)
{
	addIOption("minSurfOverlap", 60, "minimum surface overlapping for merging");
	addBOption("iterative", false, "merge the sub-windows in one pass or iteratively");
	addBOption("onlySurfOverlaps", false, "keep only sub-windows that overlap with at least another one");
	addBOption("onlyMaxSurf", false, "keep only the merged sub-window with the maximum surface");
	addBOption("onlyMaxConf", false, "keep only the merged sub-window with the maximum confidence");
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

void OverlapSelector::init(const PatternList& patterns, int minSurfOverlap, bool ignoreInclusion)
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

		info.reset(minSurfOverlap);
		for (int j = 0; j < n_patterns; j ++)
			if (i != j)
		{
			info.add(source, m_patternBuffer[j], j, ignoreInclusion);
		}
	}
}

/////////////////////////////////////////////////////////////////////////
// Add a new pattern, if it overlaps enough

void OverlapSelector::OverlapInfo::add(const Pattern& source, const Pattern& test_patt, int index_patt,
					bool ignoreInclusion)
{
	const int surf = source.getOverlap(test_patt, ignoreInclusion);
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

bool OverlapSelector::process(const PatternList& candidates)
{
	const bool verbose = getBOption("verbose");
	const bool iterative = getBOption("iterative");
	const int minSurfOverlap = getInRange(getIOption("minSurfOverlap"), 0, 100);
	const bool onlySurfOverlaps = getBOption("onlySurfOverlaps");
	const bool onlyMaxSurf = getBOption("onlyMaxSurf");
	const bool onlyMaxConf = onlyMaxSurf == false && getBOption("onlyMaxConf");

	// Check parameters
	if (m_merger == 0)
	{
		Torch::message("OverlapSelector::process - no merger specified!\n");
		return false;
	}
	if (candidates.isEmpty() == true)
	{
		if (verbose == true)
		{
			Torch::message("OverlapSelector::process - the pattern space is empty!\n");
		}
		return true;
	}

	Pattern merged_pattern;
	PatternList lpatterns;

	// Initialize the buffers (sort a copy descending, build the overlapping table ...)
	init(candidates, minSurfOverlap, true);
	if (verbose == true)
	{
		printOverlapTable();
	}

	///////////////////////////////////////////////////////////////////////////
	// Merge detections ... until all the candidates are covered
	//	(inclusion is ignored)
	///////////////////////////////////////////////////////////////////////////

	int n_steps = 0;
	bool stop = false;
	while (stop == false)
	{
		n_steps ++;
		if (verbose == true)
		{
			print("OverlapSelector: step no. [%d] ...\n", n_steps);
		}

		// Delete the old merged patterns (if any)
		lpatterns.clear();

		// Merge the overlapping detections
		int n_merged_patterns = 0;
		for (int i = 0; i < m_n_patterns; i ++)
			if (	// Check if the SW was already merged
				m_patternTags[i] == false
				&&
				// Check option condition
				(onlySurfOverlaps == false || n_steps > 1 || m_overlappingTable[i].m_n_overlaps > 0))
		{
			// Tag the detection, it should not be considered one more time!
			m_patternTags[i] = true;
			n_merged_patterns ++;

			OverlapInfo& info = m_overlappingTable[i];
			if (info.m_n_overlaps == 0)
			{
				// Nothing to merge!
				lpatterns.add(m_patternBuffer[i]);
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
				m_merger->merge(merged_pattern);

				lpatterns.add(merged_pattern);
			}
		}

		if (verbose == true)
		{
			print("\t[%d/%d] merged patterns\n", n_merged_patterns, m_n_patterns);
		}

		// Re-initialize the buffers (as above, but now from the temporary list)
		init(lpatterns, minSurfOverlap, true);
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

	///////////////////////////////////////////////////////////////////////////
	// Prune the merged SWs,
	//	check if any intersection (with inclusion too)
	//	-> choose the ones with the maximum activation!
	///////////////////////////////////////////////////////////////////////////

	stop = false;
	while (stop == false)
	{
		// Initialize the overlapping table (with the inclusion test)
		init(lpatterns, minSurfOverlap / 10, false);	// Very little intersection + inclusion

		// Delete the old merged patterns (if any)
		lpatterns.clear();

		// Prune the overlapping detections ...
		int n_pruned_patterns = 0;
		for (int i = 0; i < m_n_patterns; i ++)
			if (	m_patternTags[i] == false)
		{
			OverlapInfo& info = m_overlappingTable[i];

			if (info.m_n_overlaps == 0)
			{
				// No intersection, just add it!
				lpatterns.add(m_patternBuffer[i]);
				m_patternTags[i] = true;
			}
			else
			{
				const int crt_activation = m_patternBuffer[i].m_activation;

				// Check if it's the one with the maximum activation from the list of the overlapping ones!
				bool its_max = true;
				for (int j = 0; j < info.m_n_overlaps; j ++)
				{
					const int k = info.m_overlaps[j].m_index;
					if (m_patternBuffer[k].m_activation > crt_activation)
					{
						its_max = false;
						break;
					}
				}

				// Keep it, if yes!
				if (true == its_max)
				{
					m_patternTags[i] = true;
					for (int j = 0; j < info.m_n_overlaps; j ++)
					{
						const int k = info.m_overlaps[j].m_index;
						m_patternTags[k] = true;
					}

					lpatterns.add(m_patternBuffer[i]);
					n_pruned_patterns ++;
				}
			}
		}

		// Check if more patterns must be pruned
		stop = n_pruned_patterns == 0;
	}

	// Keep only the merged sub-window with the maximum surface
	if (onlyMaxSurf == true && lpatterns.isEmpty() == false)
	{
		Pattern max_surf_pattern;

		const int n_patterns = lpatterns.size();
		for (int i = 0; i < n_patterns; i ++)
		{
			const Pattern& pattern = lpatterns.get(i);
			if (pattern.m_w * pattern.m_h > max_surf_pattern.m_w * max_surf_pattern.m_h)
			{
				max_surf_pattern.copy(pattern);
			}
		}

		lpatterns.clear();
		lpatterns.add(max_surf_pattern);
	}

	// Keep only the merged sub-window with the maximum confidence
	else if (onlyMaxConf == true && lpatterns.isEmpty() == false)
	{
		Pattern max_conf_pattern;
		max_conf_pattern.m_confidence = -100000.0;

		const int n_patterns = lpatterns.size();
		for (int i = 0; i < n_patterns; i ++)
		{
			const Pattern& pattern = lpatterns.get(i);
			if (pattern.m_confidence > max_conf_pattern.m_confidence)
			{
				max_conf_pattern.copy(pattern);
			}
		}

		lpatterns.clear();
		lpatterns.add(max_conf_pattern);
	}

	// Accumulate the detections to the result
	m_patterns.add(lpatterns);

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////

}

