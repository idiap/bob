/**
 * @file cxx/old/scanning/src/Explorer.cc
 * @date Wed Apr 6 14:49:40 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "scanning/Explorer.h"
#include "scanning/ScaleExplorer.h"
#include "scanning/ipSWPruner.h"
#include "scanning/ipSWEvaluator.h"
#include "ip/Image.h"

namespace Torch
{

/////////////////////////////////////////////////////////////////////////
// Constructor

ExplorerData::ExplorerData(ipSWEvaluator* swEvaluator)
	:	m_swEvaluator(swEvaluator),
		m_swPruners(0), m_nSWPruners(0),
		m_image_w(0), m_image_h(0),
		m_stat_scanned(0), m_stat_prunned(0), m_stat_accepted(0),
		m_patterns()
{
}

/////////////////////////////////////////////////////////////////////////
// Destructor

ExplorerData::~ExplorerData()
{
	delete[] m_swPruners;
}

/////////////////////////////////////////////////////////////////////////
// Delete all the detections so far

void ExplorerData::clear()
{
	m_stat_scanned = 0;
	m_stat_prunned = 0;
	m_stat_accepted = 0;
	m_patterns.clear();
}

/////////////////////////////////////////////////////////////////////////
// Initialize the (evaluator + pruners) processing for these tensors

bool ExplorerData::init(const Tensor& input_prune, const Tensor& input_evaluation)
{
        for (int i = 0; i < m_nSWPruners; i ++)
        {
                if (m_swPruners[i]->process(input_prune) == false)
                {
                        return false;
                }
        }
        return m_swEvaluator->process(input_evaluation);
}

/////////////////////////////////////////////////////////////////////////
// Stores some new pattern

void ExplorerData::storePattern(const Pattern& p)
{
	return storePattern(p.m_x, p.m_y, p.m_w, p.m_h, p.m_confidence);
}

/////////////////////////////////////////////////////////////////////////
// Constructor

Explorer::Explorer()
	: 	m_data(0),
		m_scales(0),
		m_scale_explorers(0),
		m_scale_ips(0),
		m_scale_rois(0),
		m_n_scales(0)
{
	// Set the default parameters
	addIOption("min_patt_w", 0, "pattern minimum allowed width");
	addIOption("max_patt_w", 4096, "pattern maximum allowed width");
	addIOption("min_patt_h", 0, "pattern minimum allowed height");
	addIOption("max_patt_h", 4096, "pattern maximum allowed height");
	addFOption("ds", 1.25f, "scale variation from the smallest to the largest window size");
	addBOption("StopAtFirstDetection", false, "stop at the first candidate patterns");
	addBOption("StartWithLargeScales", false, "large to small scales scanning");
}

/////////////////////////////////////////////////////////////////////////
// Destructor

Explorer::~Explorer()
{
	// Cleanup allocated memory
	delete m_data;
	deallocateScales();
}

/////////////////////////////////////////////////////////////////////////
// Deallocate the scale information

void Explorer::deallocateScales()
{
	delete[] m_scales;
	delete[] m_scale_explorers;
	delete[] m_scale_ips;
	delete[] m_scale_rois;

	m_scales = 0;
	m_scale_explorers = 0;
	m_scale_ips = 0;
	m_scale_rois = 0;
	m_n_scales = 0;
}

/////////////////////////////////////////////////////////////////////////
// Resize the scale information

bool Explorer::resizeScales(int n_scales)
{
	if (n_scales < 0)
	{
		return false;
	}

	// Deallocate old scales
	deallocateScales();

	// Allocate new scales
	m_n_scales = n_scales;
	m_scales = new sSize[n_scales];
	m_scale_explorers = new ScaleExplorer*[n_scales];
	m_scale_ips = new spCore*[n_scales];
	m_scale_rois = new sRect2D[n_scales];

	// Initialize scales
	for (int i = 0; i < n_scales; i ++)
	{
		m_scale_explorers[i] = 0;
		m_scale_ips[i] = 0;
	}

	return true;
}

/////////////////////////////////////////////////////////////////////////
// Set the scanning strategies for the scales (different or the same)

bool Explorer::setScaleExplorer(ScaleExplorer* scaleExplorer)
{
	// Check parameters
	if (	m_n_scales < 1 || m_scales == 0 || scaleExplorer == 0)
	{
		Torch::message("Explorer::setScaleExplorer - invalid parameters!\n");
		return false;
	}

	// Copy the scale explorer
	for (int i = 0; i < m_n_scales; i ++)
	{
		m_scale_explorers[i] = scaleExplorer;
	}
	return true;
}

bool Explorer::setScaleExplorer(int index_scale, ScaleExplorer* scaleExplorer)
{
	// Check parameters
	if (	m_n_scales < 1 || m_scales == 0 || scaleExplorer == 0 ||
		index_scale < 0 || index_scale >= m_n_scales)
	{
		Torch::message("Explorer::setScaleExplorer - invalid parameters!\n");
		return false;
	}

	// Copy the scale explorer
	m_scale_explorers[index_scale] = scaleExplorer;
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Set the features to use for the scales (different or the same) (for pattern evaluation)
// (If they are 0/NULL, then the original input tensor will be used!)

bool Explorer::setScaleEvaluationIp(spCore* scaleEvaluationIp)
{
	// Check parameters
	if (	m_n_scales < 1 || m_scales == 0)
	{
		Torch::message("Explorer::setScaleEvaluationIp - invalid parameters!\n");
		return false;
	}

	// Copy the scale evaluator
	for (int i = 0; i < m_n_scales; i ++)
	{
		m_scale_ips[i] = scaleEvaluationIp;
	}
	return true;
}

bool Explorer::setScaleEvaluationIp(int index_scale, spCore* scaleEvaluationIp)
{
	// Check parameters
	if (	m_n_scales < 1 || m_scales == 0 ||
		index_scale < 0 || index_scale >= m_n_scales)
	{
		Torch::message("Explorer::setScaleEvaluationIp - invalid parameters!\n");
		return false;
	}

	// Copy the scale pruner
	m_scale_ips[index_scale] = scaleEvaluationIp;
	return true;
}

/////////////////////////////////////////////////////////////////
// Delete old detections (if any)

void Explorer::clear()
{
	m_data->clear();
}

/////////////////////////////////////////////////////////////////
// Initialize the scanning process with the given image size

bool Explorer::init(int image_w, int image_h)
{
	// Check if the pattern model operator is valid
	if (m_data->m_swEvaluator == 0)
	{
		Torch::message("Explorer::init - invalid pattern model operator!\n");
		return false;
	}

	// Check parameters
	if (image_w < 1 || image_h < 1)
	{
		Torch::message("Explorer::init - invalid image size!\n");
		return false;
	}

	m_data->m_image_w = image_w;
	m_data->m_image_h = image_h;

	// Clear the old detections
	m_data->clear();

	// Delete old scales (new ones should be computed by the specific implementations)
	deallocateScales();

	// NB: The scales will be computed at the XXXExplorer::init(image_w, image_h) implementation!

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Initialize the scanning process for a specific ROI

bool Explorer::init(const sRect2D& roi)
{
	// NB: The ROIs for each scale will be computed at the XXXExplorer::init(roi) implementation!

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Modify the sub-window evaluation operator (pattern model)

bool Explorer::setSWEvaluator(ipSWEvaluator* swEvaluator)
{
	// Check parameters
	if (	swEvaluator == 0)
	{
		Torch::message("Torch::Explorer::setSWEvaluator - invalid parameters!\n");
		return false;
	}

	// OK
	m_data->m_swEvaluator = swEvaluator;
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Modify the sub-window prunning operators

bool Explorer::addSWPruner(ipSWPruner* swPruner)
{
	// Check parameters
	if (	swPruner == 0)
	{
		Torch::message("Torch::Explorer::addSWPruner - invalid parameters!\n");
		return false;
	}

	// OK
	ipSWPruner** temp = new ipSWPruner*[m_data->m_nSWPruners + 1];
	for (int i = 0; i < m_data->m_nSWPruners; i ++)
	{
		temp[i] = m_data->m_swPruners[i];
	}
	delete[] m_data->m_swPruners;
	m_data->m_swPruners = temp;
	m_data->m_swPruners[m_data->m_nSWPruners ++] = swPruner;
	return true;
}

void Explorer::deleteAllSWPruners()
{
	delete[] m_data->m_swPruners;
	m_data->m_swPruners = 0;
	m_data->m_nSWPruners = 0;
}

/////////////////////////////////////////////////////////////////////////
// Access functions

int Explorer::getModelWidth() const
{
	return m_data->m_swEvaluator == 0 ? 1 : m_data->m_swEvaluator->getModelWidth();
}

int Explorer::getModelHeight() const
{
	return m_data->m_swEvaluator == 0 ? 1 : m_data->m_swEvaluator->getModelHeight();
}

const sSize& Explorer::getScale(int index_scale) const
{
	if (index_scale < 0 || index_scale >= m_n_scales)
	{
		Torch::error("Explorer::getScale - invalid scale index!\n");
	}

	return m_scales[index_scale];
}

/////////////////////////////////////////////////////////////////////////

}
