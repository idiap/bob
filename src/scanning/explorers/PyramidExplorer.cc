#include "PyramidExplorer.h"
#include "ScaleExplorer.h"
#include "ipSWEvaluator.h"
#include "Image.h"
#include "xtprobeImageFile.h"
#include "ipScaleYX.h"

namespace Torch {

/////////////////////////////////////////////////////////////////////////
// Constructor

PyramidExplorerData::PyramidExplorerData(ipSWEvaluator* swEvaluator)
	: 	ExplorerData(swEvaluator),
		m_inv_scale(1.0f)
{
}

/////////////////////////////////////////////////////////////////////////
// Destructor

PyramidExplorerData::~PyramidExplorerData()
{
}

/////////////////////////////////////////////////////////////////////////
// Store some pattern - need to be rescaled first!

void PyramidExplorerData::storePattern(int sw_x, int sw_y, int sw_w, int sw_h, double confidence)
{
        m_patterns.add(Pattern(	FixI(m_inv_scale * sw_x),
				FixI(m_inv_scale * sw_y),
				FixI(m_inv_scale * sw_w),
				FixI(m_inv_scale * sw_h),
				confidence));
}

/////////////////////////////////////////////////////////////////////////
// Set the current scanning scale

void PyramidExplorerData::setScale(const sSize& scale)
{
	m_scale = scale;
	m_inv_scale = 0.5f * (  (m_image_w + 0.0f) / (m_scale.w + 0.0f) +
                                (m_image_h + 0.0f) / (m_scale.h + 0.0f));
}

/////////////////////////////////////////////////////////////////////////
// Constructor

PyramidExplorer::PyramidExplorer(ipSWEvaluator* swEvaluator)
	: 	Explorer(swEvaluator),
                m_scaling_ips(0),
                m_scale_prune_tensors(0),
                m_scale_evaluation_tensors(0)
{
	m_data = new PyramidExplorerData(swEvaluator);

	addBOption("savePyramidsToJpg", false, "save the scaled images to JPEG");
}

/////////////////////////////////////////////////////////////////////////
// Destructor

PyramidExplorer::~PyramidExplorer()
{
        deallocateScaleTensors();
}

/////////////////////////////////////////////////////////////////////////
// Deallocate the tensors for each scale

void PyramidExplorer::deallocateScaleTensors()
{
        delete[] m_scale_prune_tensors;
        delete[] m_scale_evaluation_tensors;
        delete[] m_scaling_ips;

        m_scale_prune_tensors = 0;
        m_scale_evaluation_tensors = 0;
        m_scaling_ips = 0;
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

        // Compute the min/max and scale variance (relative to the image size)
	const float min_scale = max((model_w + 0.0f) / (max_patt_w + 0.0f), (model_h + 0.0f) / (max_patt_h + 0.0f));
	const float max_scale = min((model_w + 0.0f) / (min_patt_w + 0.0f), (model_h + 0.0f) / (min_patt_h + 0.0f));
	const float ds = getInRange(    getFOption("ds"),
                                        1.0f + 1.0f / (image_w + 0.0f),
                                        (max_patt_w + 0.0f) / (min_patt_w + 0.0f) + 2.0f / (image_w + 0.0f));

	const bool verbose = getBOption("verbose");

	// Compute the number of scales (relative to the image size)
	int n_scales = 0;
	int last_scale_w = -1;
	for (double scale = max_scale; scale >= min_scale; scale /= ds)
	{
		const int scale_w = (int)(0.5 + scale * image_w);
		if (scale_w == last_scale_w)
		{
			continue;
		}

		last_scale_w = scale_w;
		n_scales ++;
	}

	// Resize the scale information
	resizeScales(n_scales);
	deallocateScaleTensors();
	m_scale_prune_tensors = new const Tensor*[n_scales];
	m_scale_evaluation_tensors = new const Tensor*[n_scales];
	m_scaling_ips = new ipScaleYX[n_scales];

	// Compute the scales (relative to the image size)
	int i = 0;
	last_scale_w = -1;
	for (double scale = max_scale; scale >= min_scale; scale /= ds)
	{
		const int scale_w = (int)(0.5 + scale * image_w);
		if (scale_w == last_scale_w)
		{
			continue;
		}

		m_scales[i].w = (int)(0.5 + scale * image_w);
		m_scales[i].h = (int)(0.5 + scale * image_h);

		// ... debug message
		if (verbose == true)
		{
			Torch::print("[PyramidExplorer]: - generating the [%d/%d] scale: %dx%d\n",
				i + 1, m_n_scales, m_scales[i].w, m_scales[i].h);
		}

		last_scale_w = scale_w;
		i ++;
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
// Preprocess the image (extract features ...) => store data in <prune_ips> and <evaluation_ips>

bool PyramidExplorer::preprocess(const Image& image)
{
	// Check parameters
	if (m_n_scales < 1)
	{
		Torch::message("PyramidExplorer::preprocess - invalid number of scales!\n");
		return false;
	}

	const bool save_pyramids = getBOption("savePyramidsToJpg");

	// Run the <ipCore>s for prunning and evaluation for each scale
	for (int i = 0; i < m_n_scales; i ++)
	{
		const int scale_w = m_scales[i].w;
		const int scale_h = m_scales[i].h;

                ipScaleYX& ip_scale = m_scaling_ips[i];

                // Scale the image to the current scale
		if (	ip_scale.setIOption("width", scale_w) == false ||
			ip_scale.setIOption("height", scale_h) == false ||
			ip_scale.process(image) == false)
		{
			Torch::message("PyramidExplorer::preprocess - failed to run the scalling object for scale [%d/%d]!\n",
					i + 1, m_n_scales);
			return false;
		}

		// Save the scaled image
		if (save_pyramids == true)
		{
                        Image scaled_image;
                        if (    scaled_image.resize(scale_w, scale_h, image.getNPlanes()) == true &&
                                scaled_image.copyFrom(ip_scale.getOutput(0)) == true)
                        {
                                char str[200];
                                sprintf(str, "Pyramid_%dx%d.jpg", scale_w, scale_h);

                                xtprobeImageFile xtprobe;
                                xtprobe.save(scaled_image, str);
                        }
		}

		ipCore* ip_prune = m_scale_prune_ips[i];
		ipCore* ip_evaluation = m_scale_evaluation_ips[i];

		// Compute the prunning features for the scaled image
		if (ip_prune == 0)
		{
		        // The prune tensor is the scaled image!
		        m_scale_prune_tensors[i] = &ip_scale.getOutput(0);
		}
		else
		{
		        // The prune tensor uses features from the scaled image!
		        if (ip_prune->process(ip_scale.getOutput(0)) == false)
                        {
                                Torch::message("PyramidExplorer::preprocess - \
                                                failed to run the pruning <ipCore> for scale [%d/%d]!\n",
                                                i + 1, m_n_scales);
                                return false;
                        }
                        m_scale_prune_tensors[i] = &ip_prune->getOutput(0);
		}

		// Compute the evaluation features for the scaled image
		if (ip_evaluation == 0)
		{
		        // The evaluation tensor is the scaled image!
		        m_scale_evaluation_tensors[i] = &ip_scale.getOutput(0);
		}
		else
		{
		        // The evaluation tensor uses features from the scaled image!
                        if (ip_evaluation == ip_prune)// but check maybe it's the same processing!
                        {
                                m_scale_evaluation_tensors[i] = &ip_prune->getOutput(0);
                        }
                        else
                        {
                                if (ip_evaluation->process(ip_scale.getOutput(0)) == false)
                                {
                                        Torch::message("PyramidExplorer::preprocess - \
                                                        failed to run the evaluation <ipCore> for scale [%d/%d]!\n",
                                                        i + 1, m_n_scales);
                                        return false;
                                }
                                m_scale_evaluation_tensors[i] = &ip_evaluation->getOutput(0);
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

		// Initialize the evaluator/classifier for this scale
		if (m_data->init(*m_scale_prune_tensors[i], *m_scale_evaluation_tensors[i]) == false)
		{
		        Torch::message("PyramidExplorer::process - failed to initialize the pruners & classifier!\n");
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
                if (scaleExplorer->process(*m_data, stopAtFirstDetection) == false)
		{
			Torch::message("PyramidExplorer::process - failed to run scale explorer [%d/%d]!\n",
					i + 1, m_n_scales);
			return false;
		}

		// Check if we should stop at the first detection
		if (stopAtFirstDetection == true && m_data->m_patterns.isEmpty() == false)
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
	return true;
}

/////////////////////////////////////////////////////////////////////////

}
