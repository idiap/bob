#include "GreedyExplorer.h"
#include "ipSWEvaluator.h"
#include "ScaleExplorer.h"
#include "File.h"

// Context constants
static const int	NoVarX = 6;	// No. of steps on Ox
static const int	NoVarY = 6;	// No. of steps on Oy
static const int 	NoVarS = 7;	// No. of steps on scale
static const int	VarX = 5;	// %/step variation on Ox
static const int	VarY = 5;	// %/step variation on Oy
static const int	VarS = 5;	// %/step variation on scales

namespace Torch
{

/////////////////////////////////////////////////////////////////////////
// Constructor

GreedyExplorer::GreedyExplorer(Mode mode)
	: 	MSExplorer(),
		m_mode(mode),
		m_ctx_type(Full), m_ctx_size(0), m_ctx_flags(0), m_ctx_scores(0),
		m_ctx_ox(0), m_ctx_oy(0), m_ctx_os(0),
		m_ctx_sw_merger(new AveragePatternMerger)
{
	addIOption("ctx_type", 1, "0 - full context, 1 - axis context");
	optionChanged(0);
}

/////////////////////////////////////////////////////////////////////////
// Destructor

GreedyExplorer::~GreedyExplorer()
{
	delete[] m_ctx_flags;
	delete[] m_ctx_scores;
	delete[] m_ctx_ox;
	delete[] m_ctx_oy;
	delete[] m_ctx_os;
	delete m_ctx_sw_merger;
}

/////////////////////////////////////////////////////////////////////////
/// called when some option was changed

void GreedyExplorer::optionChanged(const char* name)
{
	switch (getIOption("ctx_type"))
	{
		// Axis context
	case Axis:
		if (m_ctx_type != Axis)
		{
			delete[] m_ctx_flags;
			delete[] m_ctx_scores;
			delete[] m_ctx_ox;
			delete[] m_ctx_oy;
			delete[] m_ctx_os;

			m_ctx_type = Axis;
			m_ctx_size = (2 * NoVarX + 1) + (2 * NoVarY + 1) + (2 * NoVarS + 1);

			m_ctx_flags = new unsigned char[m_ctx_size];
			m_ctx_scores = new double[m_ctx_size];
			m_ctx_ox = new int[m_ctx_size];
			m_ctx_oy = new int[m_ctx_size];
			m_ctx_os = new int[m_ctx_size];

			int index = 0;
			for (int is = -NoVarS; is <= NoVarS; is ++, index ++)
			{
				m_ctx_ox[index] = 0;
				m_ctx_oy[index] = 0;
				m_ctx_os[index] = is;
			}
			for (int ix = -NoVarX; ix <= NoVarX; ix ++, index ++)
			{
				m_ctx_ox[index] = ix;
				m_ctx_oy[index] = 0;
				m_ctx_os[index] = 0;
			}
			for (int iy = -NoVarY; iy <= NoVarY; iy ++, index ++)
			{
				m_ctx_ox[index] = 0;
				m_ctx_oy[index] = iy;
				m_ctx_os[index] = 0;
			}
		}
		break;

		// Full context
	case Full:
	default:
		if (m_ctx_type != Full)
		{
			delete[] m_ctx_flags;
			delete[] m_ctx_scores;
			delete[] m_ctx_ox;
			delete[] m_ctx_oy;
			delete[] m_ctx_os;

			m_ctx_type = Full;
			m_ctx_size = (2 * NoVarX + 1) * (2 * NoVarY + 1) * (2 * NoVarS + 1);

			m_ctx_flags = new unsigned char[m_ctx_size];
			m_ctx_scores = new double[m_ctx_size];
			m_ctx_ox = new int[m_ctx_size];
			m_ctx_oy = new int[m_ctx_size];
			m_ctx_os = new int[m_ctx_size];

			int index = 0;
			for (int is = -NoVarS; is <= NoVarS; is ++)
				for (int ix = -NoVarX; ix <= NoVarX; ix ++)
					for (int iy = -NoVarY; iy <= NoVarY; iy ++, index ++)
					{
						m_ctx_ox[index] = ix;
						m_ctx_oy[index] = iy;
						m_ctx_os[index] = is;
					}
		}
		break;
	}
}

/////////////////////////////////////////////////////////////////////////
// Set the context classifier

bool GreedyExplorer::setContextModel(const char* filename)
{
	File file;
	return 	file.open(filename, "r") &&
		m_ctx_model.loadFile(file);
}

/////////////////////////////////////////////////////////////////////////
// Process the image (check for pattern's sub-windows)

bool GreedyExplorer::process()
{
	const bool verbose = getBOption("verbose");

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
			static const int max_n_iters = 10;
			static const double eps_size = 0.05;
			static const double eps_center = 0.05;

			// TODO: how to check for convergence:
			//	- all subwindows move less than a percentage (10% of their size)
			// 	- this percentage can be made at each iteration smaller and smaller!
			// TODO: ideas to reduce the size of the context
			//	- make it smaller
			//	- sample a fixed number of random locations
			//	- sample a fixed number of learned locations
			//	- use a different 3D representation (not a grid/cube, but like a star?!)
			// NB: At the end the detections should be clustered using a relaxed kernel (constant)

			// Buffers
			Profile profile;
			DoubleTensor pf_tensor;
			Pattern sw_ctx;

			PatternList procSWs;			// Processed SWs
			PatternList crtSWs;			// Current SWs
			procSWs.add(m_data->m_patterns);

			// Greedy scanning:
			//	- MS to initialize
			//	- repeat (until convergence):
			//		- profile detections
			//		- keep only detections that pass the context-based model
			//		- move the detections to the contextual SW estimate
			int n_iters = 0;
			bool convergence = false;
			while (n_iters < max_n_iters && convergence == false)
			{
				crtSWs.clear();
				crtSWs.add(procSWs);
				procSWs.clear();

				// Test each SW against the profile model
				double max_diff_center = 0.0;
				double max_diff_size = 0.0;
				for (int i = 0; i < crtSWs.size(); i ++)
				{
					m_data->clear();

					const Pattern& sw = crtSWs.get(i);
					if (profileSW(sw) == false)
					{
						Torch::message("GreedyExplorer::process - error profiling a SW!\n");
						return false;
					}

					profile.reset(sw, *this);
					profile.copyTo(pf_tensor);

					if (m_ctx_model.forward(pf_tensor) == false)
					{
						Torch::message("GreedyExplorer::process - failed to run the profile model!\n");
						return false;
					}

					// If it's not a false alarms, replace it with its contextual estimation
					if (m_ctx_model.isPattern() == true)
					{
						m_ctx_sw_merger->merge(sw_ctx);
						procSWs.add(	sw_ctx,
								true);	// Remove duplicates!

						// Compute how much the center and the size of the SW was changed
						const double dw = abs(sw.m_w - sw_ctx.m_w);
						const double dh = abs(sw.m_h - sw_ctx.m_h);
						const double dcx = abs(sw.getCenterX() - sw_ctx.getCenterX());
						const double dcy = abs(sw.getCenterY() - sw_ctx.getCenterY());

						max_diff_size = max(max_diff_size, (dw + dh) / (sw.m_w + sw.m_h));
						max_diff_center = max(max_diff_center, (dcx + dcy) / (sw.m_w + sw.m_h));
					}
				}

				if (verbose == true)
				{
					print("GreedyExplorer: [%d/%d] - from %d to %d SWs, max_diff_size = %lf, max_diff_center = %lf ...\n",
						n_iters + 1, max_n_iters,
						crtSWs.size(), procSWs.size(),
						max_diff_size, max_diff_center);
				}

				// Check if convergence was reached
				convergence = (max_diff_size < eps_size) && (max_diff_center < eps_center);

				n_iters ++;
			}

			// Add the collected patterns to the buffer
			m_data->clear();
			for (int i = 0; i < procSWs.size(); i ++)
			{
				m_data->storePattern(procSWs.get(i));
			}
		}
		break;

		// Profiling along candidate SWs
	case Profiling:
		{
			// Nothing to do - the user retrieves the profiles using <profileSW> function!
		}
		break;
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Profile some SW - fill the profile around it

bool GreedyExplorer::profileSW(int sw_x, int sw_y, int sw_w, int sw_h)
{
	m_ctx_sw_merger->reset();
	m_ctx_sw_merger->add(Pattern(sw_x, sw_y, sw_w, sw_h, 0.0));

	const double center_x = sw_x + 0.5 * sw_w;
	const double center_y = sw_y + 0.5 * sw_h;

	int index = 0;

	switch (m_ctx_type)
	{
		// Axis context
	case Axis:
		{
			// Vary the scale
			for (int is = -NoVarS; is <= NoVarS; is ++, index ++)
			{
				const double ds = 1.0 + 0.01 * VarS * is;

				const int new_sw_w = FixI(ds * sw_w);
				const int new_sw_h = FixI(ds * sw_h);
				const int new_sw_x = FixI(center_x - 0.5 * ds * sw_w);
				const int new_sw_y = FixI(center_y - 0.5 * ds * sw_h);

				addSWToProfile(new_sw_x, new_sw_y, new_sw_w, new_sw_h, m_ctx_flags[index], m_ctx_scores[index]);
			}

			// Vary the Ox coordinate
			const int dx = max(FixI(0.01 * VarX * sw_w), 1);
			for (int ix = -NoVarX; ix <= NoVarX; ix ++, index ++)
			{
				const int new_sw_x = sw_x + ix * dx;

				addSWToProfile(new_sw_x, sw_y, sw_w, sw_h, m_ctx_flags[index], m_ctx_scores[index]);
			}

			// Vary the Oy coordinate
			const int dy = max(FixI(0.01 * VarY * sw_h), 1);
			for (int iy = -NoVarY; iy <= NoVarY; iy ++, index ++)
			{
				const int new_sw_y = sw_y + iy * dy;

				addSWToProfile(sw_x, new_sw_y, sw_w, sw_h, m_ctx_flags[index], m_ctx_scores[index]);
			}
		}
		break;

		// Full context
	case Full:
		{
			// Vary the scale
			for (int is = -NoVarS; is <= NoVarS; is ++)
			{
				const double ds = 1.0 + 0.01 * VarS * is;
				const int new_sw_w = FixI(ds * sw_w);
				const int new_sw_h = FixI(ds * sw_h);

				const int dx = max(FixI(0.01 * VarX * new_sw_w), 1);
				const int dy = max(FixI(0.01 * VarY * new_sw_h), 1);

				// Vary the Ox position
				for (int ix = -NoVarX; ix <= NoVarX; ix ++)
				{
					const int new_sw_x = sw_x + ix * dx;

					// Vary the Oy position
					for (int iy = -NoVarY; iy <= NoVarY; iy ++, index ++)
					{
						const int new_sw_y = sw_y + iy * dy;

						addSWToProfile(	new_sw_x, new_sw_y, new_sw_w, new_sw_h,
								m_ctx_flags[index], m_ctx_scores[index]);
					}
				}
			}

		}
		break;
	}

	// OK
	return true;
}

bool GreedyExplorer::profileSW(const Pattern& pattern)
{
	return profileSW(pattern.m_x, pattern.m_y, pattern.m_w, pattern.m_h);
}

/////////////////////////////////////////////////////////////////////////
// Add a subwindow to the profile

void GreedyExplorer::addSWToProfile(	int sw_x, int sw_y, int sw_w, int sw_h,
					unsigned char& flag, double& score)
{
	const int image_w = m_data->m_image_w;
	const int image_h = m_data->m_image_h;

	const int model_w = m_data->m_swEvaluator->getModelWidth();
	const int model_h = m_data->m_swEvaluator->getModelHeight();

	// Default profile: no detection, low score
	flag = 0x00;
	score = -1000.0;

	// Process the sub-window, ignore if some error
	//      (the coordinates may fall out of the image)
	if (	sw_w >= model_w && sw_h >= model_h &&
		sw_w < image_w && sw_h < image_h)
	{
		const int old_size = m_data->m_patterns.size();
		if (	ScaleExplorer::processSW(sw_x, sw_y, sw_w, sw_h, *m_data) &&
			m_data->m_patterns.size() != old_size)
		{
			const Pattern& det_sw = m_data->m_patterns.get(old_size);
			flag = 0x01;
			score = det_sw.m_confidence;
			m_ctx_sw_merger->add(det_sw);
		}
	}
}

/////////////////////////////////////////////////////////////////////////

}
