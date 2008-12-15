#include "MSExplorer.h"
#include "ScaleExplorer.h"
#include "ipSWEvaluator.h"
#include "Image.h"

namespace Torch {

/////////////////////////////////////////////////////////////////////////
// Constructor

MSExplorerData::MSExplorerData(ipSWEvaluator* swEvaluator)
	: ExplorerData(swEvaluator)
{
}

/////////////////////////////////////////////////////////////////////////
// Destructor

MSExplorerData::~MSExplorerData()
{
}

/////////////////////////////////////////////////////////////////////////
// Store some pattern - just copy it!

void MSExplorerData::storePattern(int sw_x, int sw_y, int sw_w, int sw_h, float confidence)
{
	m_patternSpace.add(Pattern(sw_x, sw_y, sw_w, sw_h, confidence));
}

/////////////////////////////////////////////////////////////////////////
// Constructor

MSExplorer::MSExplorer(ipSWEvaluator* swEvaluator)
	: 	Explorer(swEvaluator),
                m_prune_tensor(0),
                m_evaluation_itensor(0)
{
	m_data = new MSExplorerData(swEvaluator);
}

/////////////////////////////////////////////////////////////////////////
// Destructor

MSExplorer::~MSExplorer()
{
}

/////////////////////////////////////////////////////////////////////////
// Set the features to use for the scales
//	(for prunning and pattern evaluation)
// It's enforced to use the same pruneIp and evaluationIp!
//	=> the <index_scale> functions will return false!!!

bool MSExplorer::setScalePruneIp(ipCore* scalePruneIp)
{
	return Explorer::setScalePruneIp(scalePruneIp);
}

bool MSExplorer::setScalePruneIp(int index_scale, ipCore* scalePruneIp)
{
	return false;
}

bool MSExplorer::setScaleEvaluationIp(ipCore* scaleEvaluationIp)
{
	return Explorer::setScaleEvaluationIp(scaleEvaluationIp);
}

bool MSExplorer::setScaleEvaluationIp(int index_scale, ipCore* scaleEvaluationIp)
{
	return false;
}

/////////////////////////////////////////////////////////////////////////
// Initialize the scanning process with the given image size

bool MSExplorer::init(int image_w, int image_h)
{
        // Check parameters
	if (Explorer::init(image_w, image_h) == false)
	{
		return false;
	}

	const int param_min_patt_w = getIOption("min_patt_w");
	const int param_max_patt_w = getIOption("max_patt_w");
	const int param_min_patt_h = getIOption("min_patt_h");
	const int param_max_patt_h = getIOption("max_patt_h");
	if (    param_max_patt_w < param_min_patt_w ||
                param_max_patt_h < param_min_patt_h)
        {
                Torch::message("MSExplorer::init - invalid pattern size!\n");
                return false;
        }

	// Model size
	const int model_w = getModelWidth();
	const int model_h = getModelHeight();

	// Get the min/max pattern width/height
	const int min_patt_w = getInRange(param_min_patt_w, model_w, image_w);
	const int max_patt_w = getInRange(param_max_patt_w, model_w, image_w);
	const int min_patt_h = getInRange(param_min_patt_h, model_h, image_h);
	const int max_patt_h = getInRange(param_max_patt_h, model_h, image_h);

	// Compute the min/max and scale variance (relative the model size)
	const double min_scale = max((min_patt_w + 0.0) / (model_w + 0.0), (min_patt_h + 0.0) / (model_h + 0.0));
	const double max_scale = min((max_patt_w + 0.0) / (model_w + 0.0), (max_patt_h + 0.0) / (model_h + 0.0));
	const double ds = getInRange(   getFOption("ds"),
                                        1.0 + 1.0 / (model_w + 0.0),
                                        (max_patt_w + 0.0) / (min_patt_w + 0.0) + 2.0 / model_w + 0.0);

        const bool verbose = getBOption("verbose");

	// Compute the number of scales (relative to the model size)
	int n_scales = 0;
	for (double scale = min_scale; scale <= max_scale; scale *= ds, n_scales ++)
	{
	}

	// Resize the scale information
	resizeScales(n_scales);

	// Compute the scales (relative to the model size)
	int i = 0;
	for (double scale = min_scale; scale <= max_scale; scale *= ds, i ++)
	{
	        m_scales[i].w = (int)(0.5 + scale * model_w);
		m_scales[i].h = (int)(0.5 + scale * model_h);

		// ... debug message
		if (verbose == true)
		{
			Torch::print("[MSExplorer]: - generating the [%d/%d] scale: %dx%d\n",
				i + 1, m_n_scales, m_scales[i].w, m_scales[i].h);
		}
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Initialize the scanning process for a specific ROI

bool MSExplorer::init(const sRect2D& roi)
{
	if (Explorer::init(roi) == false)
	{
		return false;
	}

	// Set the same ROI for each scale
	for (int i = 0; i < m_n_scales; i ++)
	{
		m_scale_rois[i] = roi;
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Check if the scanning can continue (or the space was explored enough)

bool MSExplorer::hasMoreSteps() const
{
	return Explorer::hasMoreSteps();
}

/////////////////////////////////////////////////////////////////////////
// Preprocess the image (extract features ...) => store data in <prune_ips> and <evaluation_ips>

bool MSExplorer::preprocess(const Image& image)
{
	ipCore* ip_prune = m_scale_prune_ips[0];
	ipCore* ip_evaluation = m_scale_evaluation_ips[0];

	const Tensor* prune_tensor = 0;
	const Tensor* evaluation_tensor = 0;

	// Check parameters
	if (m_n_scales < 1)
	{
		Torch::message("MSExplorer::preprocess - invalid <ipCore>s for pruning or evaluation!\n");
		return false;
	}

	// Compute the prune features for the whole image
	if (ip_prune == 0)
	{
	        // The initial image!
                m_prune_tensor = &image;
	}
	else
	{
	        // Some features need to be extracted!
                if (	ip_prune->setInputSize(image.getWidth(), image.getHeight()) == false ||
                        ip_prune->process(image) == false)
                {
                        Torch::message("MSExplorer::preprocess - failed to run the pruning <ipCore>!\n");
                        return false;
                }
                m_prune_tensor = &ip_prune->getOutput(0);
	}

	// Compute the evaluation features for the whole image
	if (ip_evaluation == 0)
	{
	        // The initial image!
	        evaluation_tensor = &image;
	}
	else
	{
	        // Some features need to be extracted!
	        if (ip_evaluation == ip_prune)  // but check maybe it's the same processing!
	        {
	                evaluation_tensor = &ip_prune->getOutput(0);
	        }
	        else
	        {
			if (	ip_evaluation->setInputSize(image.getWidth(), image.getHeight()) == false ||
                                ip_evaluation->process(image) == false)
                        {
                                Torch::message("MSExplorer::preprocess - failed to run the evaluation <ipCore>!\n");
                                return false;
                        }
                        evaluation_tensor = &ip_evaluation->getOutput(0);
	        }
	}

	// Compute the integral image for the evaluation tensor
        if (    m_ipi_evaluation.setInputSize(image.getWidth(), image.getHeight()) == false ||
                m_ipi_evaluation.process(*evaluation_tensor) == false)
        {
                Torch::message("MSExplorer::preprocess - failed to compute the evaluation integral image!");
                return false;
        }

        // Set the final tensor for evaluation
        m_evaluation_itensor = &m_ipi_evaluation.getOutput(0);

	//OK
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Process the image (check for pattern's sub-windows)

bool MSExplorer::process()
{
	// Already finished!
	if (m_stepCounter > 0)
	{
		Torch::message("MSExplorer::process - the processing already finished!");
		return false;
	}

	// Get parameters
	const bool verbose = getBOption("verbose");
	const bool stopAtFirstDetection = getBOption("StopAtFirstDetection");
	const bool startWithLargeScales = getBOption("StartWithLargeScales");

	const int first_scale_index = startWithLargeScales == true ? m_n_scales - 1 : 0;
	const int last_scale_index = startWithLargeScales == true ? -1 : m_n_scales;
	const int delta_scale_index = startWithLargeScales == true ? -1 : 1;

	// Initialize the pruners&classifier for this scale
        if (m_data->init(*m_prune_tensor, *m_evaluation_itensor) == false)
        {
                Torch::message("MSExplorer::process - failed to initialize the pruners & classifier!\n");
                return false;
        }

	// Run for each scale the associated ScaleExplorer
	for (int i = first_scale_index; i != last_scale_index; i += delta_scale_index)
	{
		ScaleExplorer* scaleExplorer = m_scale_explorers[i];

		// Check parameters
		if (scaleExplorer == 0)
		{
			Torch::message("MSExplorer::process - invalid scale explorer [%d/%d]!\n",
					i + 1, m_n_scales);
			return false;
		}

		// Debug message
		if (verbose == true)
		{
			Torch::print("[MSExplorer]: running the scale explorer [%d/%d] for the size [%dx%d]\n",
					i + 1, m_n_scales, m_scales[i].w, m_scales[i].h);
		}

		// Initialize and run the scale explorer
		if (scaleExplorer->init(m_scales[i].w, m_scales[i].h, m_scale_rois[i]) == false)
		{
			Torch::message("MSExplorer::process - failed to initialize scale explorer [%d/%d]!\n",
					i + 1, m_n_scales);
			return false;
		}
		//      ... process the integral images of the prune and evaluation tensors
		if (scaleExplorer->process(*m_data, stopAtFirstDetection) == false)
		{
			Torch::message("MSExplorer::process - failed to run scale explorer [%d/%d]!\n",
					i + 1, m_n_scales);
			return false;
		}

		// Check if we should stop at the first detection
		if (stopAtFirstDetection == true && m_data->m_patternSpace.isEmpty() == false)
		{
			// Debug message
			if (verbose == true)
			{
				Torch::print("[MSExplorer]: stopped at the first detection!\n");
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
