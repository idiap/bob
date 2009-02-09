#include "MeanShiftSelector.h"

namespace Torch
{

/////////////////////////////////////////////////////////////////////////
// Constructor

MeanShiftSelector::MeanShiftSelector()
{
}

/////////////////////////////////////////////////////////////////////////
// Destructor

MeanShiftSelector::~MeanShiftSelector()
{
}

/////////////////////////////////////////////////////////////////////////
// Delete all stored patterns

void MeanShiftSelector::clear()
{
	Selector::clear();
}

/////////////////////////////////////////////////////////////////////////
// Constructor

MeanShiftSelector::Grid::Cell::Cell()
	:	m_indexes(0),
		m_size(0), m_capacity(0)
{
}

/////////////////////////////////////////////////////////////////////////
// Destructor

MeanShiftSelector::Grid::Cell::~Cell()
{
	delete[] m_indexes;
	m_indexes = 0;
}

/////////////////////////////////////////////////////////////////////////
// Add a new index

void MeanShiftSelector::Grid::Cell::add(int index_pattern)
{
	// Check if this index is already registered - binary search
	for (int i = 0; i < m_size; i ++)
		if (m_indexes[i] == index_pattern)
		{
			return;	// Already added!
		}

	// OK, add it
	if (m_size >= m_capacity)
	{
		resize();
	}
	m_indexes[m_size ++] = index_pattern;
}

/////////////////////////////////////////////////////////////////////////
// Resize the indexes already stored to accomodate new ones

void MeanShiftSelector::Grid::Cell::resize()
{
	const int delta = m_size * 8 / 5 + 1;

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
// Initialize the grid structure with the given points

void MeanShiftSelector::Grid::init(const PatternList& lpatterns)
{
	// Clear old indexes
	for (int i = 0; i < GridSize; i ++)
		for (int j = 0; j < GridSize; j ++)
		{
			m_cells[i][j].m_size = 0;
		}

	// Estimate the grid coordinates and cell size
	const int n_patterns = lpatterns.size();

	int x_min = 100000, x_max = 0;
	int y_min = 100000, y_max = 0;
	for (int i = 0; i < n_patterns; i ++)
	{
		const Pattern& pattern = lpatterns.get(i);

		x_min = min(x_min, pattern.m_x);
		x_max = max(x_max, pattern.m_x + pattern.m_w);

		y_min = min(y_min, pattern.m_y);
		y_max = max(y_max, pattern.m_y + pattern.m_h);
	}

	x_min --; x_max ++;
	y_min --; y_max ++;

	m_x = x_min;
	m_y = y_min;

	m_dx = FixI((x_max - x_min + 0.0f) / (GridSize + 0.0f));
	m_dy = FixI((y_max - y_min + 0.0f) / (GridSize + 0.0f));
	m_inv_dx = 1.0f / m_dx;
	m_inv_dy = 1.0f / m_dy;

	// Add each point to its cell
	for (int i = 0; i < n_patterns; i ++)
	{
		getCell(lpatterns.get(i)).add(i);
	}
}

/////////////////////////////////////////////////////////////////////////
// Print the grid structure

void MeanShiftSelector::Grid::print() const
{
	Torch::print("*** Grid partition ***************************************\n");
	for (int i = 0; i < GridSize; i ++)
		for (int j = 0; j < GridSize; j ++)
		{
			const Grid::Cell& cell = m_cells[i][j];
			if (cell.m_size > 0)
			{
				Torch::print("\t[%d/%d] hash has [%d] indexes: ", i + 1, j + 1, cell.m_size);
				for (int k = 0; k < cell.m_size; k ++)
				{
					Torch::print("%d, ", cell.m_indexes[k]);
				}
				Torch::print("\n");
			}
		}
	Torch::print("**********************************************************\n");
}

/////////////////////////////////////////////////////////////////////////
// Get the associated value for some point

MeanShiftSelector::Grid::Cell& MeanShiftSelector::Grid::getCell(const Pattern& pattern)
{
	return getCell(pattern.getCenterX(),
			pattern.getCenterY(),
			pattern.m_w,
			pattern.m_h);
}

MeanShiftSelector::Grid::Cell& MeanShiftSelector::Grid::getCell(float cx, float cy, float w, float h)
{
	const int x_index = getInRange(FixI((cx - m_x) * m_inv_dx), 0, GridSize - 1);
	const int y_index = getInRange(FixI((cy - m_y) * m_inv_dy), 0, GridSize - 1);

	return m_cells[x_index][y_index];
}

/////////////////////////////////////////////////////////////////////////
// Get the closest points to the given one (using LSH table)
// Returns the number of found points

int MeanShiftSelector::getClosest(const Pattern& pattern)
{
	return getClosest(	(float)pattern.getCenterX(),
				(float)pattern.getCenterY(),
				(float)pattern.m_w,
				(float)pattern.m_h);
}

int MeanShiftSelector::getClosest(float cx, float cy, float w, float h)
{
	const int x_index = getInRange(FixI((cx - m_grid.m_x) * m_grid.m_inv_dx), 0, GridSize - 1);
	const int y_index = getInRange(FixI((cy - m_grid.m_y) * m_grid.m_inv_dy), 0, GridSize - 1);

	// Start from the corresponding center cell and increase radius until some non-empty cell is found ...
	int isize = addClosest(m_grid.m_cells[x_index][y_index], 0);
	int radius = 1;
	do
	{
		const int x_min_index = getInRange(x_index - radius, 0, GridSize - 1);
		const int x_max_index = getInRange(x_index + radius, 0, GridSize - 1);

		const int y_min_index = getInRange(y_index - radius, 0, GridSize - 1);
		const int y_max_index = getInRange(y_index + radius, 0, GridSize - 1);

		// Add the cells having <radius> distance to the center
		for (int x = x_min_index; x <= x_max_index; x ++)
		{
			isize = addClosest(m_grid.m_cells[x][y_min_index], isize);
			isize = addClosest(m_grid.m_cells[x][y_max_index], isize);
		}
		for (int y = y_min_index + 1; y < y_max_index; y ++)
		{
			isize = addClosest(m_grid.m_cells[x_min_index][y], isize);
			isize = addClosest(m_grid.m_cells[x_max_index][y], isize);
		}

		radius ++;
	}
	while (isize < MaxNoClosestPoints && radius < GridSize / 2);

	// Return the number of closest points
	return isize;
}

int MeanShiftSelector::addClosest(const Grid::Cell& cell, int isize)
{
	for (int k = 0; k < cell.m_size && isize < MaxNoClosestPoints; k ++)
	{
		m_iclosest[isize ++] = cell.m_indexes[k];
	}
	return isize;
}

/////////////////////////////////////////////////////////////////////////
// Get the distance between two points (cx, cy, w, h)

float MeanShiftSelector::getDistance(const Pattern& pattern, float cx, float cy, float w, float h)
{
	const float diff_cx = pattern.getCenterX() - cx;
	const float diff_cy = pattern.getCenterY() - cy;
	const float diff_w = pattern.m_w - w;
	const float diff_h = pattern.m_h - h;

	return diff_cx * diff_cx + diff_cy * diff_cy + diff_w * diff_w + diff_h * diff_h;
}

float MeanShiftSelector::getDistance(	float cx1, float cy1, float w1, float h1,
					float cx2, float cy2, float w2, float h2)
{
	const float diff_cx = cx1 - cx2;
	const float diff_cy = cy1 - cy2;
	const float diff_w = w1 - w2;
	const float diff_h = h1 - h2;

	return diff_cx * diff_cx + diff_cy * diff_cy + diff_w * diff_w + diff_h * diff_h;
}

/////////////////////////////////////////////////////////////////////////
// Process the list of candidate sub-windows and select the best ones
// (this will accumulate them to the pattern list)

int compare_floats(const void* a, const void* b)
{
	const float* da = (const float*) a;
	const float* db = (const float*) b;

	return (*da > *db) - (*da < *db);
}


bool MeanShiftSelector::process(const PatternList& candidates)
{
	// Get parameters
	const bool verbose = getBOption("verbose");
	const int max_n_iters = 100;		// Maximum number of iterations
	const float kclosest_max_var = 3.0f;	// Maximum variation (avg +/- n * stdev) to find kclosest
	const float inv_kclosest_max_var = 1.0f / kclosest_max_var;

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

	// Grid structure (initialization)
	m_grid.init(candidates);
	if (verbose == true)
	{
		m_grid.print();
	}

	const int n_patterns = candidates.size();
	float* bandwidths = new float[n_patterns];
	float* inv_bandwidths = new float[n_patterns];

	srand((unsigned int)time(0));

	// AMS: compute the adaptive bandwidth for each point/pattern/SW
	for (int i = 0; i < n_patterns; i ++)
	{
		const Pattern& crt_pattern = candidates.get(i);

		// Get the closest points to estimate the bandwidth
		int n_closest_points = getClosest(crt_pattern);
		if (n_closest_points <= 1)
		{
			bandwidths[i] = 0.0f;
		}
		else
		{
			float crt_cx = crt_pattern.getCenterX();
			float crt_cy = crt_pattern.getCenterY();
			float crt_w = crt_pattern.m_w;
			float crt_h = crt_pattern.m_h;

			// Compute the distances and sort them
			static float distances[MaxNoClosestPoints];
			static float buffer[MaxNoClosestPoints];

			for (int j = 0; j < n_closest_points; j ++)
			{
				const Pattern& pattern = candidates.get(m_iclosest[j]);
				distances[j] = getDistance(pattern, crt_cx, crt_cy, crt_w, crt_h);
			}

			//const int kth = getInRange(kclosest, 1, n_closest_points);
			//bandwidths[i] = kth_element(distances, n_closest_points, kth, buffer);

			qsort(distances, n_closest_points, sizeof(float), compare_floats);

			// Choose the kclosest points as to have the given maximum variance
			float sum = distances[0] + distances[1];
			float sum_square = distances[0] * distances[0] + distances[1] * distances[1];

			int kclosest = 1;
			while (kclosest + 1 < n_closest_points)
			{
				const float inv = 1.0f / (kclosest + 1.0f);
				const float inv_square = inv * inv;

				const float diff = inv_kclosest_max_var * (distances[kclosest + 1] - inv * sum);
				if (inv * sum_square - inv_square * sum * sum < diff * diff)
				{
					break;
				}

				kclosest ++;
				sum += distances[kclosest];
				sum_square += distances[kclosest] * distances[kclosest];
			}

			// Set the bandwidth as the distance to the kclosest point
			bandwidths[i] = distances[kclosest];
			inv_bandwidths[i] = bandwidths[i] == 0.0f ? 1.0f : 1.0f / bandwidths[i];
		}
	}

	// AMS (for each point until convergence)
	int sum_n_iters = 0;
	int cnt_n_iters = 0;
	for (int i = 0; i < n_patterns; i ++)
	{
		const Pattern& crt_pattern = candidates.get(i);

		float crt_cx = crt_pattern.getCenterX();
		float crt_cy = crt_pattern.getCenterY();
		float crt_w = crt_pattern.m_w;
		float crt_h = crt_pattern.m_h;

		// Iterate until convergence (or the maximum number of iterations was reached)
		int n_iters = 0;
		int cnt = 0;
		//print("------------------------\n");
		while ((n_iters ++) < max_n_iters)
		{
			float sum_cx = 0.0f, sum_cy = 0.0f;
			float sum_w = 0.0f, sum_h = 0.0f;
			float sum_weights = 0.0f;
			cnt = 0;

			// Compute the mean shift looking for neighbour patterns
			int n_closest_points = getClosest(crt_cx, crt_cy, crt_w, crt_h);
			for (int j = 0; j < n_closest_points; j ++)
			{
				const int k = m_iclosest[j];
				const Pattern& pattern = candidates.get(k);
				const float distance = getDistance(pattern, crt_cx, crt_cy, crt_w, crt_h);
				if (distance < bandwidths[k])
				{
					const float weight = inv_bandwidths[k] * getKernel(distance, bandwidths[k]);
					sum_cx += weight * pattern.getCenterX();
					sum_cy += weight * pattern.getCenterY();
					sum_w += weight * pattern.m_w;
					sum_h += weight * pattern.m_h;
					sum_weights += weight;
					cnt ++;
				}
			}

			// Update the current position (if not converged yet)
			if (cnt <= 1)
			{
				break;
			}

			const float inv = 1.0f / sum_weights;
			const float new_crt_cx = inv * sum_cx;
			const float new_crt_cy = inv * sum_cy;
			const float new_crt_w = inv * sum_w;
			const float new_crt_h = inv * sum_h;

			static const float eps = 0.00005f;
			const float dist = getDistance(	new_crt_cx, new_crt_cy, new_crt_w, new_crt_h,
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
		m_patterns.add(Pattern(	FixI(crt_cx - 0.5f * crt_w),
					FixI(crt_cy - 0.5f * crt_h),
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
			(sum_n_iters + 0.0f) / (cnt_n_iters == 0 ? 1.0f : (cnt_n_iters + 0.0f)));
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Fast (linear time) median search (kth-element, more general)
// http://valis.cs.uiuc.edu/~sariel/research/CG/applets/linear_prog/median.html
// (std::nth_element function from STL does the same thing!!!)

float MeanShiftSelector::kth_element(float* data, int size, int kth, float* buffer)
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
	const float test_val = data[rand() % size];
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

