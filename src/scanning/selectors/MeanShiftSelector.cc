#include "MeanShiftSelector.h"

namespace Torch
{

/////////////////////////////////////////////////////////////////////////
// Constructor

MeanShiftSelector::MeanShiftSelector()
	:	m_iclosest(0), m_iDistClosest(0), m_kclosest(0),
		m_bandwidthsComputed(false), m_bandwidths(0), m_inv_bandwidths2(0), m_inv_bandwidths6(0),
		m_n_allocated(0)
{
	addIOption("kernel", 0, "0 - constant, 1 - linear, 2 - quadratic");
}

/////////////////////////////////////////////////////////////////////////
// Destructor

MeanShiftSelector::~MeanShiftSelector()
{
	delete[] m_iclosest;
	delete[] m_iDistClosest;
	delete[] m_kclosest;
	delete[] m_bandwidths;
	delete[] m_inv_bandwidths2;
	delete[] m_inv_bandwidths6;
}

/////////////////////////////////////////////////////////////////////////
// Delete all stored patterns

void MeanShiftSelector::clear()
{
	Selector::clear();
}

/////////////////////////////////////////////////////////////////////////
// Check if two sub-window intersects

static bool areSWIntersected(	double sw1_cx, double sw1_cy, double sw1_w, double sw1_h,
				double sw2_cx, double sw2_cy, double sw2_w, double sw2_h)
{
	const double sw1_x = sw1_cx - 0.5 * sw1_w;
	const double sw1_y = sw1_cy - 0.5 * sw1_h;

	const double sw2_x = sw2_cx - 0.5 * sw2_w;
	const double sw2_y = sw2_cy - 0.5 * sw2_h;

	return 	sw2_x <= sw1_x + sw1_w &&
		sw2_x + sw2_w >= sw1_x &&
		sw2_y <= sw1_y + sw1_h &&
		sw2_y + sw2_h >= sw1_y;

//	// Check the corners - left
//	const double x_min = min(sw1_x, sw2_x);
//	const double overlap_x_min = max(sw1_x, sw2_x);
//
//	// Check the corners - top
//	const double y_min = min(sw1_y, sw2_y);
//	const double overlap_y_min = max(sw1_y, sw2_y);
//
//	// Check the corners - right
//	const double x_max = max(sw1_x + sw1_w, sw2_x + sw2_w);
//	const double overlap_x_max = min(sw1_x + sw1_w, sw2_x + sw2_w);
//
//	// Check the corners - bottom
//	const double y_max = max(sw1_y + sw1_h, sw2_y + sw2_h);
//	const double overlap_y_max = min(sw1_y + sw1_h, sw2_y + sw2_h);
//
//	// No intersection
//	return  !(	overlap_x_max < overlap_x_min || overlap_y_max < overlap_y_min ||
//			x_max - x_min > sw1_w + sw2_w || y_max - y_min > sw1_h + sw2_h);
}

static bool areSWIntersected(	const Pattern& pattern,
				double sw2_cx, double sw2_cy, double sw2_w, double sw2_h)
{
	return areSWIntersected(pattern.getCenterX(), pattern.getCenterY(), pattern.m_w, pattern.m_h,
				sw2_cx, sw2_cy, sw2_w, sw2_h);
}

/////////////////////////////////////////////////////////////////////////
// Compute the Jaccard distance between two subwindows

//static double Jaccard(	double sw1_cx, double sw1_cy, double sw1_w, double sw1_h,
//			double sw2_cx, double sw2_cy, double sw2_w, double sw2_h)
//{
//	const double sw1_x = sw1_cx - 0.5 * sw1_w;
//	const double sw1_y = sw1_cy - 0.5 * sw1_h;
//
//	const double sw2_x = sw2_cx - 0.5 * sw2_w;
//	const double sw2_y = sw2_cy - 0.5 * sw2_h;
//
//	// Check for intersection
//	if (	sw2_x <= sw1_x + sw1_w &&
//		sw2_x + sw2_w >= sw1_x &&
//		sw2_y <= sw1_y + sw1_h &&
//		sw2_y + sw2_h >= sw1_y)
//	{
//		// Intersection - compute distance
//		const double x_min = max(sw1_x, sw2_x);
//		const double x_max = min(sw1_x + sw1_w, sw2_x + sw2_w);
//
//		const double y_min = max(sw1_y, sw2_y);
//		const double y_max = min(sw1_y + sw1_h, sw2_y + sw2_h);
//
//		const double inters = (x_max - x_min) * (y_max - y_min);
//
//		// http://en.wikipedia.org/wiki/Jaccard_index
//		return 1.0 - inters / (sw1_w * sw1_h + sw2_w * sw2_h - inters);
//	}
//	else
//	{
//		// No intersection
//		return 1.0;
//	}
//}
//
//static double Jaccard(	const Pattern& pattern,
//			double sw2_cx, double sw2_cy, double sw2_w, double sw2_h)
//{
//	return Jaccard(	pattern.getCenterX(), pattern.getCenterY(), pattern.m_w, pattern.m_h,
//			sw2_cx, sw2_cy, sw2_w, sw2_h);
//}

/////////////////////////////////////////////////////////////////////////
// Process the list of candidate sub-windows and select the best ones
// (this will accumulate them to the pattern list)

bool MeanShiftSelector::process(const PatternList& candidates)
{
	const bool verbose = getBOption("verbose");
	const int kernel_type = getInRange(getIOption("kernel"), 0, 2);

	FnGetKernel fn_kernels[3] =
	{
		getConstantKernel,
		getLinearKernel,
		getQuadraticKernel
	};

	FnGetKernel fn_kernel = fn_kernels[kernel_type];

	// Check parameters
	if (candidates.isEmpty() == true)
	{
		if (verbose == true)
		{
			Torch::message("MeanShiftSelector::process - the pattern space is empty!\n");
		}
		return true;
	}

//	// Cluster in two steps:
//	//	- first the width and height of the sub-windows
//	//	- second the center of the sub-windows
//	PatternList temp_patterns;
//	const double avg_iters_wh = cluster(	candidates, temp_patterns,
//						false, false, true, true,
//						getESquareDistWH, getESquareDistWH,
//						fn_kernel,
//						verbose);
//
//	const double avg_iters_cxy = cluster(	temp_patterns, m_patterns,
//						true, true, false, false,
//						getESquareDistCxCy, getESquareDistCxCy,
//						fn_kernel,
//						verbose);

	// AMS clustering
	const double avg_iters = cluster(	candidates, m_patterns,
						true, true, true, true,
						getESquareDistAll, getESquareDistAll,
						fn_kernel,
						verbose);

	// Debug
	if (verbose == true)
	{
//		print("Adaptive Mean Shift clustering - average number of iterations: step1 = %lf, step2 = %lf\n",
//			avg_iters_wh, avg_iters_cxy);
		print("Adaptive Mean Shift clustering - average number of iterations = %lf.\n",
			avg_iters);
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Estimate the bandwidth of some point using the kclosest distance
// marking the border of the variance in the (avg +/- n * stdev) domain
//		(== the k+1 point will have a too high variance!!! ==)

double MeanShiftSelector::getBandwidth(double* distances, int size, int& kclosest)
{
	static const double eps = 0.000001;

	if (size <= 1)
	{
		// An unrelated sub-window
		kclosest = 0;
		return 0.0;
	}

	else
	{
		static const double kclosest_max_var = 3.0;	// Maximum variation (avg +/- n * stdev) to find kclosest
		static const double inv_kclosest_max_var = 1.0 / kclosest_max_var;

		// Choose the kclosest points as to have the given maximum variance
		double sum = distances[0] + distances[1];
		double sum_square = distances[0] * distances[0] + distances[1] * distances[1];

		// Search the closest points that are within average +/- N * sigma from the previous ones
		kclosest = 1;
		while (kclosest + 1 < size)
		{
			const double inv = 1.0 / (kclosest + 1.0);
			const double average = inv * sum;
			const double diff = inv_kclosest_max_var * (distances[kclosest + 1] - average);

			if (	distances[kclosest] > eps &&
				sum_square - average * average < diff * diff * kclosest)
			{
				break;
			}

			kclosest ++;
			sum += distances[kclosest];
			sum_square += distances[kclosest] * distances[kclosest];
		}

		return distances[kclosest];
	}
}

/////////////////////////////////////////////////////////////////////////
// Cluster using AMS (in 2 dimensions) the given subwindows
// - returns the average number of iterations

double MeanShiftSelector::cluster(	const PatternList& candidates, PatternList& clusters,
					bool mask_cx, bool mask_cy, bool mask_w, bool mask_h,
					FnGetESquareDist fn_esquare, FnGetESquareDist_P fn_esquare_p,
					FnGetKernel fn_kernel,
					bool verbose)
{
	static const double eps = 0.000001;

	////////////////////////////////////////////////////////////////
	// Cluster the patterns using AMS (Adaptive Bandwidth Mean Shift)

	// Allocate buffers
	const int n_patterns = candidates.size();
	if (n_patterns > m_n_allocated)
	{
		delete[] m_iclosest;
		delete[] m_iDistClosest;
		delete[] m_kclosest;
		delete[] m_bandwidths;
		delete[] m_inv_bandwidths2;
		delete[] m_inv_bandwidths6;

		m_n_allocated = n_patterns;

		m_iclosest = new int[m_n_allocated * MaxNoClosestPoints];
		m_iDistClosest = new double[m_n_allocated * MaxNoClosestPoints];
		m_kclosest = new int[m_n_allocated];
		m_bandwidths = new double[m_n_allocated];
		m_inv_bandwidths2 = new double[m_n_allocated];
		m_inv_bandwidths6 = new double[m_n_allocated];
	}

	// AMS: initialize bandwidths as the kclosest distance with an intersected subwindow
	m_bandwidthsComputed = false;
	for (int i = 0; i < n_patterns; i ++)
	{
		int* idx_closest = &m_iclosest[i * MaxNoClosestPoints];
		double* dist_closest = &m_iDistClosest[i * MaxNoClosestPoints];

		const int n_closest = getClosest(	candidates, candidates.get(i),
							idx_closest, dist_closest,
							fn_esquare_p);

		m_bandwidths[i] = getBandwidth(dist_closest, n_closest, m_kclosest[i]);
	}

	// AMS: adjust the bandwidths
	bool changed = true;
	while (changed == true)
	{
		changed = false;
		for (int i = 0; i < n_patterns; i ++)
		{
			int* idx_closest = &m_iclosest[i * MaxNoClosestPoints];
			double* dist_closest = &m_iDistClosest[i * MaxNoClosestPoints];
			const int kclosest = m_kclosest[i];

			// An unrelated sub-window
			if (kclosest == 0)
			{
				continue;
			}

			// Relax the bandwidth if there is with kclosest subwindow having the bandwidth
			//	smaller than the distance to the current point
			// => implies that the current point is not correlated with some of the kclosest points
			for (int k = 0; k < kclosest; k ++)
			{
				const int j = idx_closest[k];
				if (m_bandwidths[j] + eps < dist_closest[k])
				{
					m_kclosest[i] = kclosest - 1;
					m_bandwidths[i] = dist_closest[kclosest - 1];
					changed = true;
					break;
				}
			}
		}
	}
	m_bandwidthsComputed = true;

	// AMS: precompute the inverses of the bandwidths
	for (int i = 0; i < n_patterns; i ++)
	{
		m_bandwidths[i] += 1.0;
		m_inv_bandwidths2[i] = 1.0 / m_bandwidths[i];
		m_inv_bandwidths6[i] = m_inv_bandwidths2[i] * m_inv_bandwidths2[i] * m_inv_bandwidths2[i];
	}

	// AMS: converge each point to its cluster
	int sum_n_iters = 0;
	int cnt_n_iters = 0;
	srand((unsigned int)time(0));
	for (int i = 0; i < n_patterns; i ++)
	{
		// Find its stationary point
		const Pattern& crt_pattern = candidates.get(i);
		double crt_cx = crt_pattern.getCenterX();
		double crt_cy = crt_pattern.getCenterY();
		double crt_w = crt_pattern.m_w;
		double crt_h = crt_pattern.m_h;
		const int n_iters = converge(	crt_cx, crt_cy, crt_w, crt_h,
						mask_cx, mask_cy, mask_w, mask_h,
						candidates,
						fn_esquare, fn_esquare_p, fn_kernel);

		// Add the converged point (~ density mode) to the list
		clusters.add(Pattern(	FixI(crt_cx) - FixI(crt_w) / 2,
					FixI(crt_cy) - FixI(crt_h) / 2,
					FixI(crt_w),
					FixI(crt_h),
					crt_pattern.m_confidence),
				true);	// Check duplicates!

		sum_n_iters += n_iters;
		cnt_n_iters ++;
	}

	return (sum_n_iters + 0.0) / (cnt_n_iters == 0 ? 1.0 : (cnt_n_iters + 0.0));
}

/////////////////////////////////////////////////////////////////////////
// Converge a given point using AMS
// - returns the number of iterations

int MeanShiftSelector::converge(double& crt_cx, double& crt_cy, double& crt_w, double& crt_h,
				bool mask_cx, bool mask_cy, bool mask_w, bool mask_h,
				const PatternList& candidates,
				FnGetESquareDist fn_esquare, FnGetESquareDist_P fn_esquare_p,
				FnGetKernel fn_kernel)
{
	static const double eps = 0.000001;
	const int max_n_iters = 100;

	// Iterate until convergence (or the maximum number of iterations was reached)
	int n_iters = 0;
	while ((n_iters ++) < max_n_iters)
	{
		double sum_cx = 0.0, sum_cy = 0.0;
		double sum_w = 0.0, sum_h = 0.0;
		double sum_weights = 0.0;

		// Compute the mean shift looking for neighbour patterns
		const int n_closest = getClosest(	candidates, crt_cx, crt_cy, crt_w, crt_h,
							m_iclosest, m_iDistClosest,
							fn_esquare_p);
		for (int j = 0; j < n_closest; j ++)
		{
			const int k = m_iclosest[j];
			const Pattern& pattern = candidates.get(k);

			const double distance = m_iDistClosest[j];
			const double weight = m_inv_bandwidths6[k] * (*fn_kernel)(distance, m_inv_bandwidths2[k]);

			sum_cx += weight * pattern.getCenterX();
			sum_cy += weight * pattern.getCenterY();
			sum_w += weight * pattern.m_w;
			sum_h += weight * pattern.m_h;
			sum_weights += weight;
		}

		// Update the current position (if not converged already)
		if (n_closest <= 1)
		{
			break;
		}

		const double inv = 1.0 / sum_weights;
		const double new_crt_cx = mask_cx == true ? inv * sum_cx : crt_cx;
		const double new_crt_cy = mask_cy == true ? inv * sum_cy : crt_cy;
		const double new_crt_w = mask_w == true ? inv * sum_w : crt_w;
		const double new_crt_h = mask_h == true ? inv * sum_h : crt_h;

		if ((*fn_esquare)(	new_crt_cx, new_crt_cy, new_crt_w, new_crt_h,
					crt_cx, crt_cy, crt_w, crt_h) < eps)
		{
			// Convergence: the new sub-window is overlapping almost perfectly the one at the last iteration!
			break;
		}

		crt_cx = new_crt_cx;
		crt_cy = new_crt_cy;
		crt_w = new_crt_w;
		crt_h = new_crt_h;
	}

	return n_iters;
}

/////////////////////////////////////////////////////////////////////////
// Get the closest points to the given one
// - returns the number of found points

int MeanShiftSelector::getClosest(	const PatternList& candidates,
					double cx, double cy, double w, double h,
					int* idx_closest, double* dist_closest,
					FnGetESquareDist_P fn_esquare_p)
{
	const int n_patterns = candidates.size();

	int isize = 0;
	if (m_bandwidthsComputed == false)
	{
		// Add to the list each sub-window with at least some intersection!
		for (int i = 0; i < n_patterns && isize < MaxNoClosestPoints; i ++)
			if (areSWIntersected(candidates.get(i), cx, cy, w, h) == true)
			{
				idx_closest[isize] = i;
				dist_closest[isize ++] = (*fn_esquare_p)(candidates.get(i), cx, cy, w, h);
			}

		// Sort the distances keeping the indexes
		for (int i = 0; i < isize; i ++)
			for (int j = i + 1; j < isize; j ++)
				if (dist_closest[i] > dist_closest[j])
				{
					double temp_distance = dist_closest[i];
					dist_closest[i] = dist_closest[j];
					dist_closest[j] = temp_distance;

					int temp_index = idx_closest[i];
					idx_closest[i] = idx_closest[j];
					idx_closest[j] = temp_index;
				}
	}
	else
	{
		// Add to the list each sub-window within its bandwidth!
		for (int i = 0; i < n_patterns && isize < MaxNoClosestPoints; i ++)
			if (areSWIntersected(candidates.get(i), cx, cy, w, h) == true)
			{
				const double distance = (*fn_esquare_p)(candidates.get(i), cx, cy, w, h);
				if (distance < m_bandwidths[i])
				{
					idx_closest[isize] = i;
					dist_closest[isize ++] = distance;
				}
			}
	}

	return isize;
}

/////////////////////////////////////////////////////////////////////////

}

