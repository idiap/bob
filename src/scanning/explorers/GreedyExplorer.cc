#include "GreedyExplorer.h"
#include "ipSWEvaluator.h"
#include "ScaleExplorer.h"
#include "File.h"

namespace Torch
{

/////////////////////////////////////////////////////////////////////////
// Constructor

GreedyExplorer::GreedyExplorer(Mode mode)
	: 	MSExplorer(),
		m_mode(mode),
		m_profileFlags(new unsigned char[NoConfigs]),
		m_profileScores(new double[NoConfigs]),
		m_sampleOxCoefs(new double[NoConfigs]),
		m_sampleOyCoefs(new double[NoConfigs]),
		m_sampleOsCoefs(new double[NoConfigs])
{
	addIOption("sampling", 0, "0 - linear, 1 - quadratic, 2 - cubic, 3 - exponential(2), 4 - exponential(4)");
}

/////////////////////////////////////////////////////////////////////////
// Destructor

GreedyExplorer::~GreedyExplorer()
{
	delete[] m_profileFlags;
	delete[] m_profileScores;
	delete[] m_sampleOxCoefs;
	delete[] m_sampleOyCoefs;
	delete[] m_sampleOsCoefs;
}

/////////////////////////////////////////////////////////////////////////
// called when some option was changed - overriden

static double getSign(int value)
{
	return value < 0 ? -1.0 : 1.0;
}

void GreedyExplorer::optionChanged(const char* name)
{
	switch (getIOption("sampling"))
	{
	case 0:	// Linear
		{
			int index = 0;
			for (int is = -NoVarS; is <= NoVarS; is ++)
				for (int ix = -NoVarX; ix <= NoVarX; ix ++)
					for (int iy = -NoVarY; iy <= NoVarY; iy ++)
					{
						const double dx = 0.01 * VarX * ix;
						const double dy = 0.01 * VarY * iy;
						const double ds = 1.0 + 0.01 * VarS * is;
						m_sampleOxCoefs[index] = dx;
						m_sampleOyCoefs[index] = dy;
						m_sampleOsCoefs[index] = ds;
						index ++;
					}
		}
		break;

	case 1:	// Quadratic
		{
			const double norm_ox = 0.01 * VarX * NoVarX / (NoVarX * NoVarX + 0.0);
			const double norm_oy = 0.01 * VarY * NoVarY / (NoVarY * NoVarY + 0.0);
			const double norm_os = 0.01 * VarS * NoVarS / (NoVarS * NoVarS + 0.0);

			int index = 0;
			for (int is = -NoVarS; is <= NoVarS; is ++)
				for (int ix = -NoVarX; ix <= NoVarX; ix ++)
					for (int iy = -NoVarY; iy <= NoVarY; iy ++)
					{
						const double dx = norm_ox * getSign(ix) * ix * ix;
						const double dy = norm_oy * getSign(iy) * iy * iy;
						const double ds = 1.0 + norm_os * getSign(is) * is * is;
						m_sampleOxCoefs[index] = dx;
						m_sampleOyCoefs[index] = dy;
						m_sampleOsCoefs[index] = ds;
						index ++;
					}
		}
		break;

	case 2:	// Cubic
		{
			const double norm_ox = 0.01 * VarX * NoVarX / (NoVarX * NoVarX * NoVarX + 0.0);
			const double norm_oy = 0.01 * VarY * NoVarY / (NoVarY * NoVarY * NoVarY + 0.0);
			const double norm_os = 0.01 * VarS * NoVarS / (NoVarS * NoVarS * NoVarS + 0.0);

			int index = 0;
			for (int is = -NoVarS; is <= NoVarS; is ++)
				for (int ix = -NoVarX; ix <= NoVarX; ix ++)
					for (int iy = -NoVarY; iy <= NoVarY; iy ++)
					{
						const double dx = norm_ox * getSign(ix) * ix * ix * ix;
						const double dy = norm_oy * getSign(iy) * iy * iy * iy;
						const double ds = 1.0 + norm_os * getSign(is) * is * is * is;
						m_sampleOxCoefs[index] = dx;
						m_sampleOyCoefs[index] = dy;
						m_sampleOsCoefs[index] = ds;
						index ++;
					}
		}
		break;

	case 3:	// Exponential(2)
		{
			const double alpha = 2.0;	// exp(2t)

			const double norm_ox = 0.01 * VarX * NoVarX / exp(alpha * NoVarX);
			const double norm_oy = 0.01 * VarY * NoVarY / exp(alpha * NoVarY);
			const double norm_os = 0.01 * VarS * NoVarS / exp(alpha * NoVarS);

			int index = 0;
			for (int is = -NoVarS; is <= NoVarS; is ++)
				for (int ix = -NoVarX; ix <= NoVarX; ix ++)
					for (int iy = -NoVarY; iy <= NoVarY; iy ++)
					{
						const double dx = norm_ox * getSign(ix) * exp(alpha * fabs(ix));
						const double dy = norm_oy * getSign(iy) * exp(alpha * fabs(iy));
						const double ds = 1.0 + norm_os * getSign(is) * exp(alpha * fabs(is));
						m_sampleOxCoefs[index] = dx;
						m_sampleOyCoefs[index] = dy;
						m_sampleOsCoefs[index] = ds;
						index ++;
					}
		}
		break;

	case 4:	// Exponential(4)
	default:
		{
			const double alpha = 4.0;	// exp(4t)

			const double norm_ox = 0.01 * VarX * NoVarX / exp(alpha * NoVarX);
			const double norm_oy = 0.01 * VarY * NoVarY / exp(alpha * NoVarY);
			const double norm_os = 0.01 * VarS * NoVarS / exp(alpha * NoVarS);

			int index = 0;
			for (int is = -NoVarS; is <= NoVarS; is ++)
				for (int ix = -NoVarX; ix <= NoVarX; ix ++)
					for (int iy = -NoVarY; iy <= NoVarY; iy ++)
					{
						const double dx = norm_ox * getSign(ix) * exp(alpha * fabs(ix));
						const double dy = norm_oy * getSign(iy) * exp(alpha * fabs(iy));
						const double ds = 1.0 + norm_os * getSign(is) * exp(alpha * fabs(is));
						m_sampleOxCoefs[index] = dx;
						m_sampleOyCoefs[index] = dy;
						m_sampleOsCoefs[index] = ds;
						index ++;
					}
		}
		break;
	}
}

/////////////////////////////////////////////////////////////////////////
// Set the profile classifier

bool GreedyExplorer::setClassifier(const char* filename)
{
	File file;
	return 	file.open(filename, "r") &&
		m_profileModel.loadFile(file);
}

/////////////////////////////////////////////////////////////////////////
// Process the image (check for pattern's sub-windows)

bool GreedyExplorer::process()
{
	//const bool verbose = getBOption("verbose");

	// Don't want to stop at the first iteration or to scale from large to small windows
	setBOption("StopAtFirstDetection", false);
	setBOption("StartWithLargeScales", false);

	// Scan the image using MS
	if (MSExplorer::process() == false)
	{
		Torch::message("GreedyExplorer::process - error initializing the search space!\n");
		return false;
	}

	// Check the working mode ...
	switch (m_mode)
	{
		// Object detection
	case Scanning:
		{
			Profile profile;
			DoubleTensor pf_tensor;

			// Cluster the SWs generated from MSExplorer
			m_clusterAlgo.clear();
			if (m_clusterAlgo.process(m_data->m_patterns) == false)
			{
				Torch::message("GreedyExplorer::process - error clustering SWs!\n");
				return false;
			}

			// Test each SW against the profile model
			const PatternList& sws = m_clusterAlgo.getPatterns();
			PatternList tempSws;
			for (int i = 0; i < sws.size(); i ++)
			{
				const Pattern& sw = sws.get(i);
				m_data->clear();
				if (profileSW(sw) == false)
				{
					Torch::message("GreedyExplorer::process - error profiling a SW!\n");
					return false;
				}

				profile.reset(sw, m_profileFlags, m_profileScores);
				profile.copyTo(pf_tensor);

				if (m_profileModel.forward(pf_tensor) == false)
				{
					Torch::message("GreedyExplorer::process - failed to run the profile model!\n");
					return false;
				}

				if (m_profileModel.isPattern() == true)
				{
					tempSws.add(sw);
				}
			}

			// Add the collected patterns to the buffer
			m_data->clear();
			for (int i = 0; i < tempSws.size(); i ++)
			{
				m_data->storePattern(tempSws.get(i));
			}

			/*
			// Refine the search around the best points til it's possible
			const int old_n_candidates = m_data->m_patterns.size();
			while (true)
			{
				if (refineSearch() == false)
				{
					Torch::message("GreedyExplorer::process - error refining the search space!\n");
					return false;
				}

				// Debug message
				if (verbose == true)
				{
					Torch::print("[GreedyExplorer]: pruned = %d, scanned = %d, accepted = %d\n",
							m_data->m_stat_prunned,
							m_data->m_stat_scanned,
							m_data->m_stat_accepted);
				}

				// Check the stopping criterion
				if (shouldSearchMode(old_n_candidates) == false)
				{
					break;

					// Debug message
					if (verbose == true)
					{
						Torch::print("[GreedyExplorer]: stopping ...\n");
					}
				}
			}
			*/
		}
		break;

		// Profiling along candidate SWs
	case Profiling:
		{
			// Nothing to do - the user should retrieves the profiles by <profileSW>!
		}
		break;
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Refine the search around the best points

bool GreedyExplorer::refineSearch()
{
	/*
	// Make a copy of the best patterns (they will be modified with each <searchAround> call)
	const int n_best_points = m_data->m_patternSpace.getNoBest();
	for (int i = 0; i < n_best_points; i ++)
	{
		m_best_patterns[i].copy(m_data->m_patternSpace.getBest(i));

	}

	// Search around each best pattern point
	for (int i = 0; i < n_best_points; i ++)
	{
		if (searchAround(m_best_patterns[i]) == false)
		{
			Torch::message("GreedyExplorer::refineSearch - error searching around some point!");
			return false;
		}
	}
	*/

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Check if the search should be stopped
//	(it is becoming too fine or no pattern found so far?!)

bool GreedyExplorer::shouldSearchMode(int old_n_candidates) const
{
	// If no pattern found so far, then the search should be stopped
	if (m_data->m_patterns.isEmpty() == true)
	{
		return false;
	}

	// If no pattern was added at the last iteration, ...
	if (m_data->m_patterns.size() == old_n_candidates)
	{
		return false;
	}

	// OK, keep on searching
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Profile some SW - fill the profile around it

bool GreedyExplorer::profileSW(int sw_x, int sw_y, int sw_w, int sw_h)
{
	const int image_w = m_data->m_image_w;
	const int image_h = m_data->m_image_h;

	const int model_w = m_data->m_swEvaluator->getModelWidth();
	const int model_h = m_data->m_swEvaluator->getModelHeight();

	// Vary the scale ...
	int index = 0;
	for (int is = -NoVarS; is <= NoVarS; is ++)
	{
		// Vary the position ...
		for (int ix = -NoVarX; ix <= NoVarX; ix ++)
		{
			for (int iy = -NoVarY; iy <= NoVarY; iy ++)
			{
				const int new_sw_w = FixI(m_sampleOsCoefs[index] * sw_w);
				const int new_sw_h = FixI(m_sampleOsCoefs[index] * sw_h);
				const int new_sw_x = sw_x + FixI(m_sampleOxCoefs[index] * sw_w);
				const int new_sw_y = sw_y + FixI(m_sampleOyCoefs[index] * sw_h);

				// Check if the subwindow's size is too large or too small
				const bool valid = 	new_sw_w >= model_w && new_sw_h >= model_h &&
							new_sw_w < image_w && new_sw_h < image_h;

				// Default profile: no detection, low score
				m_profileFlags[index] = 0x00;
				m_profileScores[index] = -1000.0;

				// Process the sub-window, ignore if some error
				//      (the coordinates may fall out of the image)
				if (valid == true)
				{
					const int old_size = m_data->m_patterns.size();
					if (	ScaleExplorer::processSW(new_sw_x, new_sw_y, new_sw_w, new_sw_h, *m_data) &&
						m_data->m_patterns.size() != old_size)
					{
						m_profileScores[index] = m_data->m_patterns.get(old_size).m_confidence;
						m_profileFlags[index] = 0x01;
					}
				}

				// Next profile
				index ++;
			}
		}
	}

	// OK
	return true;
}

bool GreedyExplorer::profileSW(const Pattern& pattern)
{
	return profileSW(pattern.m_x, pattern.m_y, pattern.m_w, pattern.m_h);
}

/////////////////////////////////////////////////////////////////////////

}
