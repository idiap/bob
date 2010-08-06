#ifndef _TORCHVISION_SCANNING_MEAN_SHIFT_SELECTOR_H_
#define _TORCHVISION_SCANNING_MEAN_SHIFT_SELECTOR_H_

#include "Selector.h"		// <MeanShiftSelector> is a <Selector>

namespace Torch
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::Scanning::MeanShiftSelector
	//	- merges and selects the final pattern sub-windows using the Mean Shift
	//		clustering algorithm
	//
	//      - PARAMETERS (name, type, default value, description):
        //		"kernel"	int	0	"0 - constant, 1 - linear, 2 - quadratic"
	//
        // TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class MeanShiftSelector : public Selector
	{
	public:

		// Constructor
		MeanShiftSelector();

		// Destructor
		virtual ~MeanShiftSelector();

		// Delete all stored patterns
		virtual void		clear();

		// Process the list of candidate sub-windows and select the best ones
		// (this will accumulate them to the pattern list)
		virtual bool		process(const PatternList& candidates);

	private:

		/////////////////////////////////////////////////////////////////

		// Square 2D Euclidean distance
		static double		getESquareDist(double x1, double y1, double x2, double y2)
		{
			const double diff_x = x1 - x2;
			const double diff_y = y1 - y2;

			return diff_x * diff_x + diff_y * diff_y;
		}

		// Square Euclidean distances between two sub-windows
		static double		getESquareDistCxCy(	double sw1_cx, double sw1_cy, double sw1_w, double sw1_h,
								double sw2_cx, double sw2_cy, double sw2_w, double sw2_h)
		{
			return getESquareDist(sw1_cx, sw1_cy, sw2_cx, sw2_cy);
		}
		static double		getESquareDistCxCy(const Pattern& pattern, double cx, double cy, double w, double h)
		{
			return getESquareDist(pattern.getCenterX(), pattern.getCenterY(), cx, cy);
		}

		static double		getESquareDistWH(	double sw1_cx, double sw1_cy, double sw1_w, double sw1_h,
								double sw2_cx, double sw2_cy, double sw2_w, double sw2_h)
		{
			return getESquareDist(sw1_w, sw1_h, sw2_w, sw2_h);
		}
		static double		getESquareDistWH(const Pattern& pattern, double cx, double cy, double w, double h)
		{
			return getESquareDist(pattern.m_w, pattern.m_h, w, h);
		}

		static double		getESquareDistAll(	double sw1_cx, double sw1_cy, double sw1_w, double sw1_h,
								double sw2_cx, double sw2_cy, double sw2_w, double sw2_h)
		{
			return 	getESquareDist(sw1_cx, sw1_cy, sw2_cx, sw2_cy) +
				getESquareDist(sw1_w, sw1_h, sw2_w, sw2_h);
		}
		static double		getESquareDistAll(const Pattern& pattern, double cx, double cy, double w, double h)
		{
			return getESquareDistAll(	pattern.getCenterX(), pattern.getCenterY(), pattern.m_w, pattern.m_h,
							cx, cy, w, h);
		}

		typedef double (*FnGetESquareDist) (double, double, double, double, double, double, double, double);
		typedef double (*FnGetESquareDist_P) (const Pattern&, double, double, double, double);

		// Kernel functions (using the square Euclidean distance and the inverse of the bandwidth)
		static double		getConstantKernel(double e_distance, double e_inv_bandwidth)
		{
			return 1.0;
		}
		static double		getLinearKernel(double e_distance, double e_inv_bandwidth)
		{
			const double prod = 1.0 - e_distance * e_inv_bandwidth;
			return prod;
		}
		static double		getQuadraticKernel(double e_distance, double e_inv_bandwidth)
		{
			const double prod = 1.0 - e_distance * e_inv_bandwidth;
			return prod * prod;
		}

		typedef double (*FnGetKernel)(double, double);

		// Cluster using AMS (in 2 dimensions) the given subwindows
		// - returns the average number of iterations
		double			cluster(const PatternList& candidates, PatternList& clusters,
						bool mask_cx, bool mask_cy, bool mask_w, bool mask_h,
						FnGetESquareDist fn_esquare, FnGetESquareDist_P fn_esquare_p,
						FnGetKernel fn_kernel,
						bool verbose);

		// Get the closest points to the given one
		// - returns the number of found points
		int			getClosest(	const PatternList& candidates,
							const Pattern& pattern,
							int* idx_closest, double* dist_closest,
							FnGetESquareDist_P fn_esquare_p)
		{
			return getClosest(	candidates,
						pattern.getCenterX(), pattern.getCenterY(), pattern.m_w, pattern.m_h,
						idx_closest, dist_closest,
						fn_esquare_p);
		}
		int			getClosest(	const PatternList& candidates,
							double cx, double cy, double w, double h,
							int* idx_closest, double* dist_closest,
							FnGetESquareDist_P fn_esquare_p);

		// Estimate the bandwidth of some point using the kclosest distance
		// marking the border of the variance in the (avg +/- n * stdev) domain
		//		(== the k+1 point will have a too high variance!!! ==)
		double			getBandwidth(double* distances, int size, int& kclosest);

		// Converge a given point using AMS
		// - returns the number of iterations
		int			converge(	double& crt_cx, double& crt_cy, double& crt_w, double& crt_h,
							bool mask_cx, bool mask_cy, bool mask_w, bool mask_h,
							const PatternList& candidates,
							FnGetESquareDist fn_esquare, FnGetESquareDist_P fn_esquare_p,
							FnGetKernel fn_kernel);

		/////////////////////////////////////////////////////////////////
		// Attributes

		// Buffers for the indexes and the distances of the closest points
		static const int 	MaxNoClosestPoints = 128;
		int* 			m_iclosest;		// [No. subwindows] x [MaxNoClosestPoints]
		double*			m_iDistClosest;		// [No. subwindows] x [MaxNoClosestPoints]
		int*			m_kclosest;		// [No. subwindows]

		// Buffers for the bandwidths
		bool			m_bandwidthsComputed;
		double*			m_bandwidths;		// [No. subwindows] Square Euclidean distance
		double*			m_inv_bandwidths2;	// [No. subwindows] 1 / Square Euclidean distance
		double*			m_inv_bandwidths6;	// [No. subwindows] 1 / power 6 of the Euclidean distance

		// Number of allocated subwindows
		int			m_n_allocated;
	};
}

#endif
