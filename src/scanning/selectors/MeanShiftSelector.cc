#include "MeanShiftSelector.h"

namespace Torch
{

/////////////////////////////////////////////////////////////////////////
// Constructor

MeanShiftSelector::MeanShiftSelector()
	:	m_bandwidthsComputed(false),
		m_bandwidths(0),
		m_inv_bandwidths(0),
		m_candidates(0)
{
}

/////////////////////////////////////////////////////////////////////////
// Destructor

MeanShiftSelector::~MeanShiftSelector()
{
	delete[] m_bandwidths;
	delete[] m_inv_bandwidths;
}

/////////////////////////////////////////////////////////////////////////
// Delete all stored patterns

void MeanShiftSelector::clear()
{
	Selector::clear();
}

/////////////////////////////////////////////////////////////////////////
// Get the closest points to the given one (using LSH table)
// Returns the number of found points

inline int MeanShiftSelector::getClosest(const Pattern& pattern)
{
	return getClosest(	(double)pattern.getCenterX(),
				(double)pattern.getCenterY(),
				(double)pattern.m_w,
				(double)pattern.m_h);
}

int MeanShiftSelector::getClosest(double cx, double cy, double w, double h)
{
	const int n_patterns = m_candidates->size();
	int isize = 0;

	if (m_bandwidthsComputed == false)
	{
		// Add to the list each sub-window with at least some intersection!
		for (int i = 0; i < n_patterns && isize < MaxNoClosestPoints; i ++)
		{
			const double distance = getDistance(m_candidates->get(i), cx, cy, w, h);
			if (distance < 1.0)
			{
				m_iclosest[isize] = i;
				m_iDistClosest[isize ++] = distance;
			}
		}
	}
	else
	{
		// Add to the list each sub-window within its bandwidth!
		for (int i = 0; i < n_patterns && isize < MaxNoClosestPoints; i ++)
		{
			const double distance = getDistance(m_candidates->get(i), cx, cy, w, h);
			if (distance < m_bandwidths[i])
			{
				m_iclosest[isize] = i;
				m_iDistClosest[isize ++] = distance;
			}
		}
	}

	return isize;
}

/////////////////////////////////////////////////////////////////////////
// Get the distance between two points (cx, cy, w, h)

static double sw_intersection(	double sw1_x, double sw1_y, double sw1_w, double sw1_h,
				double sw2_x, double sw2_y, double sw2_w, double sw2_h)
{
	// Check for intersection
	if (	sw2_x <= sw1_x + sw1_w &&
		sw2_x + sw2_w >= sw1_x &&
		sw2_y <= sw1_y + sw1_h &&
		sw2_y + sw2_h >= sw1_y)
	{
		// Intersection - compute distance
		const double x_min = max(sw1_x, sw2_x);
		const double x_max = min(sw1_x + sw1_w, sw2_x + sw2_w);

		const double y_min = max(sw1_y, sw2_y);
		const double y_max = min(sw1_y + sw1_h, sw2_y + sw2_h);

		return (x_max - x_min) * (y_max - y_min);
	}
	else
	{
		// No intersection
		return 0.0;
	}
}

static double sw_distance(	double sw1_x, double sw1_y, double sw1_w, double sw1_h,
				double sw2_x, double sw2_y, double sw2_w, double sw2_h)
{
	// http://en.wikipedia.org/wiki/Jaccard_index
	const double inters = sw_intersection(sw1_x, sw1_y, sw1_w, sw1_h, sw2_x, sw2_y, sw2_w, sw2_h);
	return 1.0 - inters / (sw1_w * sw1_h + sw2_w * sw2_h - inters);
}

inline double MeanShiftSelector::getDistance(const Pattern& pattern, double cx, double cy, double w, double h)
{
	return getDistance(	(double)pattern.getCenterX(), (double)pattern.getCenterY(),
				(double)pattern.m_w, (double)pattern.m_h,
				cx, cy, w, h);
}
inline double MeanShiftSelector::getDistance(	double cx1, double cy1, double w1, double h1,
						double cx2, double cy2, double w2, double h2)
{
	return sw_distance(	cx1 - 0.5 * w1, cy1 - 0.5 * h1, w1, h1,
				cx2 - 0.5 * w2, cy2 - 0.5 * h2, w2, h2);
}

/////////////////////////////////////////////////////////////////////////
// Process the list of candidate sub-windows and select the best ones
// (this will accumulate them to the pattern list)

bool MeanShiftSelector::process(const PatternList& candidates)
{
	// Get parameters
	const bool verbose = getBOption("verbose");
	const int max_n_iters = 100;		// Maximum number of iterations
	const double kclosest_max_var = 3.0;	// Maximum variation (avg +/- n * stdev) to find kclosest
	const double inv_kclosest_max_var = 1.0 / kclosest_max_var;

	// Check parameters
	if (candidates.isEmpty() == true)
	{
		if (verbose == true)
		{
			Torch::message("MeanShiftSelector::process - the pattern space is empty!\n");
		}
		return true;
	}

	////////////////////////////////////////////////////////////////
	// Cluster the patterns using AMS (Adaptive Bandwidth Mean Shift)
	//	and a grid structure to fast retrieve the closest points

	// Allocate bandwidths and other buffers
	const int n_patterns = candidates.size();
	delete[] m_bandwidths;
	delete[] m_inv_bandwidths;
	m_bandwidths = new double[n_patterns];
	m_inv_bandwidths = new double[n_patterns];
	m_bandwidthsComputed = false;

	for (int i = 0; i < n_patterns; i ++)
	{
		m_bandwidths[i] = 10000000.0;
	}

	m_candidates = &candidates;

	// AMS: compute the adaptive bandwidth for each point/pattern/SW
	for (int i = 0; i < n_patterns; i ++)
	{
		const Pattern& crt_pattern = candidates.get(i);

		// Get the closest points to estimate the bandwidth
		int n_closest_points = getClosest(crt_pattern);
		if (n_closest_points <= 1)
		{
			m_bandwidths[i] = 0.0;
			m_inv_bandwidths[i] = 1.0;
		}
		else
		{
			// Sort them after the distance
			qsort(m_iDistClosest, n_closest_points, sizeof(double), compare_doubles);

//			print("DISTANCES: ");
//			for (int j = 0; j < n_closest_points; j ++)
//			{
//				print("%4.3f, ", m_iDistClosest[j]);
//			}
//			print("\n");

			// Choose the kclosest points as to have the given maximum variance
			double sum = m_iDistClosest[0] + m_iDistClosest[1];
			double sum_square = m_iDistClosest[0] * m_iDistClosest[0] + m_iDistClosest[1] * m_iDistClosest[1];

			int kclosest = 1;
			if (m_iDistClosest[kclosest] > 0.99)
			{
				// If the closest point has no intersection, than the bandwidth should be zero!
				kclosest = 0;
			}
			else
			{
				// Search the closest points that are within average +/- 3 * sigma from the previous ones
				while (	kclosest + 1 < n_closest_points &&
					m_iDistClosest[kclosest + 1] < 0.99)
				{
					const double inv = 1.0 / (kclosest + 1.0);
					const double inv_square = inv * inv;

					const double diff = inv_kclosest_max_var * (m_iDistClosest[kclosest + 1] - inv * sum);
					if (inv * sum_square - inv_square * sum * sum < diff * diff)
					{
						break;
					}

					kclosest ++;
					sum += m_iDistClosest[kclosest];
					sum_square += m_iDistClosest[kclosest] * m_iDistClosest[kclosest];
				}
			}

			// Set the bandwidth as the distance to the kclosest point
			m_bandwidths[i] = m_iDistClosest[kclosest];
			m_inv_bandwidths[i] = m_bandwidths[i] == 0.0 ? 1.0 : 1.0 / pow(m_bandwidths[i], 6.0);
//			print("[%d]: bandwidth = %f, inv_bandwidth = %f\n", i, m_bandwidths[i], m_inv_bandwidths[i]);
//			print("\n");
		}
	}

	m_bandwidthsComputed = true;

	// AMS (for each point until convergence)
	int sum_n_iters = 0;
	int cnt_n_iters = 0;
	for (int i = 0; i < n_patterns; i ++)
	{
		// Uncorrelated subwindow
		if (m_bandwidths[i] == 0.0)
		{
			m_patterns.add(candidates.get(i));
			continue;
		}

		// Need to find the subwindow's cluster
		const Pattern& crt_pattern = candidates.get(i);
		double crt_cx = crt_pattern.getCenterX();
		double crt_cy = crt_pattern.getCenterY();
		double crt_w = crt_pattern.m_w;
		double crt_h = crt_pattern.m_h;

		// Iterate until convergence (or the maximum number of iterations was reached)
		int n_iters = 0;
		int cnt = 0;
		//print("------------------------\n");
		while ((n_iters ++) < max_n_iters)
		{
			double sum_cx = 0.0, sum_cy = 0.0;
			double sum_w = 0.0, sum_h = 0.0;
			double sum_weights = 0.0;
			cnt = 0;

			// Compute the mean shift looking for neighbour patterns
			const int n_closest_points = getClosest(crt_cx, crt_cy, crt_w, crt_h);
			for (int j = 0; j < n_closest_points; j ++)
			{
				const int k = m_iclosest[j];
				const double distance = m_iDistClosest[j];

				const double weight = m_inv_bandwidths[k] * getKernel(distance, m_bandwidths[k]);
				const Pattern& pattern = candidates.get(k);

				sum_cx += weight * pattern.getCenterX();
				sum_cy += weight * pattern.getCenterY();
				sum_w += weight * pattern.m_w;
				sum_h += weight * pattern.m_h;
				sum_weights += weight;

				cnt ++;
			}

			// Update the current position (if not converged yet)
			if (cnt <= 1)
			{
				break;
			}

			const double inv = 1.0 / sum_weights;
			const double new_crt_cx = inv * sum_cx;
			const double new_crt_cy = inv * sum_cy;
			const double new_crt_w = inv * sum_w;
			const double new_crt_h = inv * sum_h;

			static const double eps = 0.000001;
			const double dist = getDistance(new_crt_cx, new_crt_cy, new_crt_w, new_crt_h,
							crt_cx, crt_cy, crt_w, crt_h);
			//print("\t>>> dist = %f, cnt = %d\n", dist, cnt);
			if (dist < eps)
			{
				break;
			}

			crt_cx = new_crt_cx;
			crt_cy = new_crt_cy;
			crt_w = new_crt_w;
			crt_h = new_crt_h;
		}

		// Add the converged point (~ density mode) to the list
		m_patterns.add(Pattern(	FixI(crt_cx - 0.5 * crt_w),
					FixI(crt_cy - 0.5 * crt_h),
					FixI(crt_w),
					FixI(crt_h),
					crt_pattern.m_confidence),
				true);	// Check duplicates!

		sum_n_iters += n_iters;
		cnt_n_iters ++;

		//print("[(%d, %d) - %dx%d] --->>> [(%d, %d) - %dx%d]\n",
		//	crt_pattern.getCenterX(), crt_pattern.getCenterY(),
		//	crt_pattern.m_w, crt_pattern.m_h,
		//	FixI(crt_cx - 0.5f * crt_w),
		//	FixI(crt_cy - 0.5f * crt_h),
		//	FixI(crt_w),
		//	FixI(crt_h));
	}

	// Debug
	if (verbose == true)
	{
		print("Adaptive Mean Shift clustering: average number of iterations = %f\n",
			(sum_n_iters + 0.0) / (cnt_n_iters == 0 ? 1.0 : (cnt_n_iters + 0.0)));
	}

	// Cleanup
	delete[] m_bandwidths;
	delete[] m_inv_bandwidths;
	m_bandwidths = 0;
	m_inv_bandwidths = 0;

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Fast (linear time) median search (kth-element, more general)
// http://valis.cs.uiuc.edu/~sariel/research/CG/applets/linear_prog/median.html
// (std::nth_element function from STL does the same thing!!!)

static double kth_element(double* data, int size, int kth, double* buffer)
{
	// Trivial cases: 1 or 2 elements
	if (size == 1)
	{
		return data[0];
	}
	else if (size == 2)
	{
		return kth == 0 ? min(data[0], data[1]) : max(data[0], data[1]);
	}

	// Put the values less than <test_val> at the begining and greater at the end
	const double test_val = data[rand() % size];
	int iless = 0, igreater = 0, iequal = 0;
	for (int i = 0; i < size; i ++)
	{
		if (data[i] < test_val)
		{
			buffer[iless ++] = data[i];
		}
		else if (data[i] > test_val)
		{
			buffer[size - (1 + igreater ++)] = data[i];
		}
		else
		{
			iequal ++;
		}
	}

	// Decision
	if (iless <= kth && kth <= iless + iequal)
	{
		// Found it!
		return test_val;
	}
	else if (iless < kth)
	{
		// Search in the list of the greater values
		return kth_element(&buffer[iless + iequal], igreater, kth - (iless + iequal), data);
	}
	else
	{
		// Search in the list of the smaller values
		return kth_element(buffer, iless, kth, data);
	}
}

/////////////////////////////////////////////////////////////////////////

}

