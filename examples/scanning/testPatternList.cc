#include "Pattern.h"
#include "general.h"
#include <cassert>
#include <cstdlib>

using namespace Torch;

//////////////////////////////////////////////////////////////////////////
// Build some pattern given the index p
//////////////////////////////////////////////////////////////////////////

void buildPattern(Pattern& pattern, int p)
{
	pattern.m_x = p;
	pattern.m_y = p + 1;
	pattern.m_w = 2 * p + 3;
	pattern.m_h = p + 4;

	static const float inv = 1.0f / 1024.0f;
	pattern.m_confidence = ((rand() % 1024 - 512) + 0.0f) * inv;
}

//////////////////////////////////////////////////////////////////////////
// Check if some retrieved pattern is build with the index p
//////////////////////////////////////////////////////////////////////////

void checkPattern(const Pattern& pattern, int p)
{
	assert(pattern.m_x == p);
	assert(pattern.m_y == p + 1);
	assert(pattern.m_w == 2 * p + 3);
	assert(pattern.m_h == p + 4);
}

//////////////////////////////////////////////////////////////////////////
// Get the best stored patterns and check they are ordered
//////////////////////////////////////////////////////////////////////////

void checkBestPatterns(const PatternList& pattlist)
{
	const int n_best = pattlist.getNoBest();
	assert(n_best > 0);

	float last_confidence = 0.0f;
	for (int i = 0; i < n_best; i ++)
	{
		const Pattern& pattern = pattlist.getBest(i);
		if (i > 0)
		{
			assert(pattern.m_confidence <= last_confidence);
		}

		last_confidence = pattern.m_confidence;
	}
}

//////////////////////////////////////////////////////////////////////////
// Main
//////////////////////////////////////////////////////////////////////////

int main()
{
	const int n_tests = 100;
	const int n_patts_per_test = 1000000;
	const int n_best_patterns = 128;

	PatternList pattlist;
	assert(pattlist.size() == 0);
	assert(pattlist.isEmpty() == true);

	Pattern pattern;

	srand((unsigned int)time(0));

	// Do the tests ...
	for (int t = 0; t < n_tests; t ++)
	{
		// Invalidate any patterns already stored
		pattlist.clear();
		assert(pattlist.size() == 0);
		assert(pattlist.isEmpty() == true);

		// Add some patterns (and check they are actually added)
		for (int p = 0; p < n_patts_per_test; p ++)
		{
			buildPattern(pattern, p);
			checkPattern(pattlist.add(pattern), p);
		}
		assert(pattlist.size() == n_patts_per_test);

		// Now retrieve them and check that we have the same patterns
		assert(pattlist.isEmpty() == false);
		if (pattlist.isEmpty() == false)
		{
			for (int p = 0; p < n_patts_per_test; p ++)
			{
				checkPattern(pattlist.get(p), p);
			}

			assert(pattlist.size() == n_patts_per_test);
		}

		print("\tPASSED add/get: [%d/%d]\r", t + 1, n_tests);
	}
	print("\n");

	// Do the tests ...
	for (int t = 0; t < n_tests; t ++)
	{
		// Invalidate any patterns already stored
		pattlist.clear();
		assert(pattlist.size() == 0);
		assert(pattlist.isEmpty() == true);

		// Set the pattern list to keep track of the best patterns
		assert(pattlist.resetBestPatterns(true, n_best_patterns) == true);
		assert(pattlist.getMaxNoBest() == n_best_patterns);
		assert(pattlist.getNoBest() == 0);

		// Add some patterns (and check they are actually added)
		for (int p = 0; p < n_patts_per_test; p ++)
		{
			buildPattern(pattern, p);
			checkPattern(pattlist.add(pattern), p);
		}
		assert(pattlist.size() == n_patts_per_test);

		// Now retrieve the best ones and check they are ordered
		checkBestPatterns(pattlist);

		print("\tPASSED best:    [%d/%d]\r", t + 1, n_tests);
	}

	print("\nOK\n");

	return 0;
}

