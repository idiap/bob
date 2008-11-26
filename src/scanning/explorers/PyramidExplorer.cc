#include "PyramidExplorer.h"
#include "ScaleExplorer.h"
#include "ipCore.h"
#include "Image.h"
#include "ipScaleYX.h"

namespace Torch {

/////////////////////////////////////////////////////////////////////////
// Constructor

PyramidExplorerData::PyramidExplorerData(ipSWEvaluator* swEvaluator)
	: 	ExplorerData(swEvaluator),
		m_inv_scale_w(1.0f),
		m_inv_scale_h(1.0f)
{
}

/////////////////////////////////////////////////////////////////////////
// Destructor

PyramidExplorerData::~PyramidExplorerData()
{
}

/////////////////////////////////////////////////////////////////////////
// Store some pattern - need to be rescaled first!

void PyramidExplorerData::storePattern(int sw_x, int sw_y, int sw_w, int sw_h, float confidence)
{
	m_patternSpace.add(Pattern(	(int)(0.5f + m_inv_scale_w * sw_x * m_image_w),
					(int)(0.5f + m_inv_scale_h * sw_y * m_image_h),
					(int)(0.5f + m_inv_scale_w * sw_w * m_image_w),
					(int)(0.5f + m_inv_scale_h * sw_h * m_image_h),
					confidence));
}

/////////////////////////////////////////////////////////////////////////
// Set the current scanning scale

void PyramidExplorerData::setScale(const sSize& scale)
{
	m_scale = scale;
	m_inv_scale_w = 1.0f / (m_scale.w + 0.0f);
	m_inv_scale_h = 1.0f / (m_scale.h + 0.0f);
}

/////////////////////////////////////////////////////////////////////////
// Constructor

PyramidExplorer::PyramidExplorer(ipSWEvaluator* swEvaluator)
	: 	Explorer(swEvaluator)
{
	m_data = new PyramidExplorerData(swEvaluator);
}

/////////////////////////////////////////////////////////////////////////
// Destructor

PyramidExplorer::~PyramidExplorer()
{
}

/////////////////////////////////////////////////////////////////////////
// Set the features to use for the scales
//	(for prunning and pattern evaluation)
// It's enforced to use one pruneIp and evaluationIp per scale!
//	=> the functions without <index_scale> will return false!!!

bool PyramidExplorer::setScalePruneIp(ipCore* scalePruneIp)
{
	return false;
}

bool PyramidExplorer::setScalePruneIp(int index_scale, ipCore* scalePruneIp)
{
	return Explorer::setScalePruneIp(index_scale, scalePruneIp);
}

bool PyramidExplorer::setScaleEvaluationIp(ipCore* scaleEvaluationIp)
{
	return false;
}

bool PyramidExplorer::setScaleEvaluationIp(int index_scale, ipCore* scaleEvaluationIp)
{
	return Explorer::setScaleEvaluationIp(index_scale, scaleEvaluationIp);
}

/////////////////////////////////////////////////////////////////////////
// Initialize the scanning process with the given image size

bool PyramidExplorer::init(int image_w, int image_h)
{
	if (Explorer::init(image_w, image_h) == false)
	{
		return false;
	}

	// Model size
	const int model_w = getModelWidth();
	const int model_h = getModelHeight();

	// Get the min/max pattern width/height
	const int min_patt_w = getInRange(getIOption("min_patt_w"), model_w, image_w);
	const int max_patt_w = getInRange(getIOption("max_patt_w"), model_w, image_w);
	const int min_patt_h = getInRange(getIOption("min_patt_h"), model_h, image_h);
	const int max_patt_h = getInRange(getIOption("max_patt_h"), model_h, image_h);

	// Compute the min/max scale (variance)
	const float min_scale = max((min_patt_w + 0.0f) / (model_w + 0.0f), (min_patt_h + 0.0f) / (model_h + 0.0f));
	const float max_scale = min((max_patt_w + 0.0f) / (model_w + 0.0f), (max_patt_h + 0.0f) / (model_h + 0.0f));
	const float ds = getInRange(getFOption("ds"), 1.0f / (min(model_w, model_h) + 0.0f), 1.0f);

	const bool verbose = getBOption("verbose");

	// Compute the number of scales
	int n_scales = 0;
	for (float scale = min_scale; scale < max_scale; scale += ds, n_scales ++)
	{
	}

	// Resize the scale information
	resizeScales(n_scales);

	// Compute the scales
	int i = 0;
	for (float scale = min_scale; scale < max_scale; scale += ds, i ++)
	{
		m_scales[i].w = (int)(0.5f + scale * model_w);
		m_scales[i].h = (int)(0.5f + scale * model_h);

		// ... debug message
		if (verbose == true)
		{
			Torch::print("[PyramidExplorer]: - generating the [%d/%d] scale: %dx%d\n",
				i + 1, m_n_scales, m_scales[i].w, m_scales[i].h);
		}
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Initialize the scanning process for a specific ROI

bool PyramidExplorer::init(const sRect2D& roi)
{
	if (Explorer::init(roi) == false)
	{
		return false;
	}

	const float inv_image_w = 1.0f / (m_data->m_image_w + 0.0f);
	const float inv_image_h = 1.0f / (m_data->m_image_h + 0.0f);

	// Compute the ROI for each scale
	for (int i = 0; i < m_n_scales; i ++)
	{
		const sSize& scale = m_scales[i];

		// --- the image to scan is scaled => the ROI need to be rescaled too !!!
		m_scale_rois[i].x = (int)(0.5f + inv_image_w * (roi.x * scale.w));
		m_scale_rois[i].y = (int)(0.5f + inv_image_h * (roi.y * scale.h));
		m_scale_rois[i].w = (int)(0.5f + inv_image_w * (roi.w * scale.w));
		m_scale_rois[i].h = (int)(0.5f + inv_image_h * (roi.h * scale.h));
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Check if the scanning can continue (or the space was explored enough)

bool PyramidExplorer::hasMoreSteps() const
{
	return Explorer::hasMoreSteps();
}

/////////////////////////////////////////////////////////////////////////
// Preprocess the image (extract features ...) => store data in <prune_ips> and <evaluation_ips>

bool PyramidExplorer::preprocess(const Image& image)
{
	// Check parameters
	if (m_n_scales < 1)
	{
		Torch::message("PyramidExplorer::preprocess - invalid number of scales!\n");
		return false;
	}

	// Initialize the scalling object
	ipScaleYX ip_scale;
	if (ip_scale.setInputSize(image.getWidth(), image.getHeight()) == false)
	{
		Torch::message("PyramidExplorer::preprocess - failed to initialize scalling object!\n");
		return false;
	}

	// Run the <ipCore>s for prunning and evaluation for each scale
	for (int i = 0; i < m_n_scales; i ++)
	{
		const int scale_w = m_scales[i].w;
		const int scale_h = m_scales[i].h;

		ipCore* ip_prune = m_scale_prune_ips[i];
		ipCore* ip_evaluation = m_scale_evaluation_ips[i];

		if (ip_prune == 0 || ip_evaluation == 0)
		{
			Torch::message("PyramidExplorer::preprocess - invalid <ipCore> for scale [%d/%d]!\n",
					i + 1, m_n_scales);
			return false;
		}

		// Scale the image to the current scale
		if (	ip_scale.setOutputSize(scale_w, scale_h) == false ||
			ip_scale.process(image) == false)
		{
			Torch::message("PyramidExplorer::preprocess - failed to run the scalling object for scale [%d/%d]!\n",
					i + 1, m_n_scales);
			return false;
		}

		// Run the prune <ipCore> on the scaled image
		if (	ip_prune->setInputSize(scale_w, scale_h) == false ||
			ip_prune->process(ip_scale.getOutput(0)) == false)
		{
			Torch::message("PyramidExplorer::preprocess - \
					failed to run the pruning <ipCore> for scale [%d/%d]!\n",
					i + 1, m_n_scales);
			return false;
		}

		// Run the evaluation <ipCore> on the scaled image
		if (ip_evaluation != ip_prune)	// but check maybe it's the same processing!
		{
			if (	ip_evaluation->setInputSize(scale_w, scale_h) == false ||
				ip_evaluation->process(ip_scale.getOutput(0)) == false)
			{
				Torch::message("PyramidExplorer::preprocess - \
						failed to run the evaluation <ipCore> for scale [%d/%d]!\n",
						i + 1, m_n_scales);
				return false;
			}
		}
	}

	//OK
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Process the image (check for pattern's sub-windows)

bool PyramidExplorer::process()
{
	// Already finished!
	if (m_stepCounter > 0)
	{
		Torch::message("PyramidExplorer::process - the processing already finished!");
		return false;
	}

	const int model_w = getModelWidth();
	const int model_h = getModelHeight();

	// Get parameters
	const bool verbose = getBOption("verbose");
	const bool stopAtFirstDetection = getBOption("StopAtFirstDetection");
	const bool startWithLargeScales = getBOption("StartWithLargeScales");

	const int first_scale_index = startWithLargeScales == true ? m_n_scales - 1 : 0;
	const int last_scale_index = startWithLargeScales == true ? -1 : m_n_scales;
	const int delta_scale_index = startWithLargeScales == true ? -1 : 1;

	// Run for each scale the associated ScaleExplorer
	for (int i = first_scale_index; i != last_scale_index; i += delta_scale_index)
	{
		ScaleExplorer* scaleExplorer = m_scale_explorers[i];

		// Initialize the <ExplorerData> to the current scale
		//	(this will make sure the candidate patterns are rescaled when stored)
		((PyramidExplorerData*)m_data)->setScale(m_scales[i]);

		// Check parameters
		if (scaleExplorer == 0)
		{
			Torch::message("PyramidExplorer::process - invalid scale explorer [%d/%d]!\n",
					i + 1, m_n_scales);
			return false;
		}

		// Debug message
		if (verbose == true)
		{
			Torch::print("[PyramidExplorer]: running the scale explorer [%d/%d] for the size [%dx%d]\n",
					i + 1, m_n_scales, m_scales[i].w, m_scales[i].h);
		}

		// Initialize and run the scale explorer (the scanning window's size is the model size!!!)
		if (scaleExplorer->init(model_w, model_h, m_scale_rois[i]) == false)
		{
			Torch::message("PyramidExplorer::process - failed to initialize scale explorer [%d/%d]!\n",
					i + 1, m_n_scales);
			return false;
		}
		//	!!! make sure to give the preprocessed scaled image to the <ScaleExplorer> !!!
		if (scaleExplorer->process(	m_scale_prune_ips[i]->getOutput(0),
						m_scale_evaluation_ips[i]->getOutput(0),
						*m_data,
						stopAtFirstDetection) == false)
		{
			Torch::message("PyramidExplorer::process - failed to run scale explorer [%d/%d]!\n",
					i + 1, m_n_scales);
			return false;
		}

		// Check if we should stop at the first detection
		if (stopAtFirstDetection == true && m_data->m_patternSpace.isEmpty() == false)
		{
			// Debug message
			if (verbose == true)
			{
				Torch::print("[PyramidExplorer]: stopped at the first detection!\n");
			}

			break;
		}
	}

	// OK
	m_stepCounter ++;
	return true;
}

/////////////////////////////////////////////////////////////////////////

}
