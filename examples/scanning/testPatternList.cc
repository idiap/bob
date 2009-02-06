#include "Pattern.h"
#include "general.h"
#include <cCHECK_FATAL>
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
	CHECK_FATAL(pattern.m_x == p);
	CHECK_FATAL(pattern.m_y == p + 1);
	CHECK_FATAL(pattern.m_w == 2 * p + 3);
	CHECK_FATAL(pattern.m_h == p + 4);
}

//////////////////////////////////////////////////////////////////////////
// Main
//////////////////////////////////////////////////////////////////////////

int main()
{
	const int n_tests = 100;
	const int n_patts_per_test = 1000000;

	PatternList pattlist;
	CHECK_FATAL(pattlist.size() == 0);
	CHECK_FATAL(pattlist.isEmpty() == true);

	Pattern pattern;

	srand((unsigned int)time(0));

	// Do the tests ...
	for (int t = 0; t < n_tests; t ++)
	{
		// Invalidate any patterns already stored
		pattlist.clear();
		CHECK_FATAL(pattlist.size() == 0);
		CHECK_FATAL(pattlist.isEmpty() == true);

		// Add some patterns (and check they are actually added)
		for (int p = 0; p < n_patts_per_test; p ++)
		{
			buildPattern(pattern, p);
			checkPattern(pattlist.add(pattern), p);
		}
		CHECK_FATAL(pattlist.size() == n_patts_per_test);

		// Now retrieve them and check that we have the same patterns
		CHECK_FATAL(pattlist.isEmpty() == false);
		if (pattlist.isEmpty() == false)
		{
			for (int p = 0; p < n_patts_per_test; p ++)
			{
				checkPattern(pattlist.get(p), p);
			}

			CHECK_FATAL(pattlist.size() == n_patts_per_test);
		}

		print("\tPASSED add/get: [%d/%d]\r", t + 1, n_tests);
	}
	print("\nOK\n");

	return 0;
}

