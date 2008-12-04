#include "Scanner.h"
#include "Image.h"
#include "Explorer.h"
#include "Selector.h"

namespace Torch
{

/////////////////////////////////////////////////////////////////////////
// Constructor

Scanner::Scanner(Explorer* explorer, Selector* selector)
	: 	m_explorer(explorer),
		m_selector(selector),
		m_rois(0), m_n_rois(0),
		m_stat_scanned(0), m_stat_prunned(0), m_stat_accepted(0)
{
}

//////////////////0///////////////////////////////////////////////////////
// Destructor

Scanner::~Scanner()
{
	delete[] m_rois;
}

/////////////////////////////////////////////////////////////////////////
// Modify the ROIs

bool Scanner::addROI(const sRect2D& roi)
{
	return addROI(roi.x, roi.y, roi.w, roi.h);
}

bool Scanner::addROI(int x, int y, int w, int h)
{
	// Check parameters
	if (!	(x >= 0 && w > 0 && y >= 0 && h > 0))
	{
		Torch::message("Scanner::addROI - invalid parameters!\n");
		return false;
	}

	// OK
	sRect2D* temp = new sRect2D[m_n_rois + 1];
	for (int i = 0; i < m_n_rois; i ++)
	{
		temp[i] = m_rois[i];
	}
	delete[] m_rois;
	m_rois = temp;

	m_rois[m_n_rois].x = x;
	m_rois[m_n_rois].y = y;
	m_rois[m_n_rois].w = w;
	m_rois[m_n_rois].h = h;

	m_n_rois ++;
	return true;
}

bool Scanner::deleteROI(int index)
{
	// Check parameters
	if (	index < 0 || index >= getNoROIs())
	{
		Torch::message("Scanner::deleteROI - invalid parameters!\n");
		return false;
	}

	// OK
	sRect2D* temp = new sRect2D[m_n_rois - 1];
	for (int i = 0; i < m_n_rois; i ++)
	{
		temp[i >= index ? i - 1 : i] = m_rois[i];
	}
	delete[] m_rois;
	m_rois = temp;
	m_n_rois --;
	return true;
}

void Scanner::deleteAllROIs()
{
	delete[] m_rois;
	m_n_rois = 0;
}

/////////////////////////////////////////////////////////////////
// Change the explorer

bool Scanner::setExplorer(Explorer* explorer)
{
	if (explorer == 0)
	{
		Torch::message("Scanner::setExplorer - invalid parameters!\n");
		return false;
	}

	// OK
	m_explorer = explorer;
	return true;
}

/////////////////////////////////////////////////////////////////
// Change the selector

bool Scanner::setSelector(Selector* selector)
{
	if (selector == 0)
	{
		Torch::message("Scanner::setSelector - invalid parameters!\n");
		return false;
	}

	// OK
	m_selector = selector;
	return true;
}

/////////////////////////////////////////////////////////////////
// Initialize the scanning (check parameters/objects, initialize explorer)

bool Scanner::init(const Image& image)
{
	// Check if any explorer/selector was set
	if (m_explorer == 0 || m_selector == 0)
	{
		Torch::message("Scanner::process - invalid explorer or selector!\n");
		return false;
	}

	const int image_w = image.getWidth();
	const int image_h = image.getHeight();
	const int model_w = m_explorer->getModelWidth();
	const int model_h = m_explorer->getModelHeight();

	// Check if the model parameters are valid
	if (	model_w > image_w || model_h > image_h ||
		model_w < 1 || model_h < 1)
	{
		Torch::message("Scanner::init - invalid model/image size!\n");
		return false;
	}

	// Initialize the explorer
	if (m_explorer->init(image_w, image_h) == false)
	{
		Torch::message("Scanner::init - failed to initialize explorer!\n");
		return false;
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////
// Process some image to scan for patterns

bool Scanner::process(const Image& image)
{
	// Check if any preprocesing/explorer/selector was set
	if (m_explorer == 0 || m_selector == 0)
	{
		Torch::message("Scanner::process - invalid explorer or selector!\n");
		return false;
	}

	// Reset detections
	m_stat_scanned = 0;
	m_stat_prunned = 0;
	m_stat_accepted = 0;
	m_selector->clear();

	// If no ROI specified, set the whole image as the single ROI
	const int image_w = image.getWidth();
	const int image_h = image.getHeight();
	if (m_rois == 0)
	{
		m_rois = new sRect2D[1];
		m_rois[0].x = 0;
		m_rois[0].y = 0;
		m_rois[0].w = image_w;
		m_rois[0].h = image_h;
		m_n_rois = 1;
	}

	// Check parameters first
	if (checkParameters(image) == false)
	{
		Torch::message("Scanner::process - invalid parameters!\n");
		return false;
	}

	const bool verbose = getBOption("verbose");

	// Preprocess the image
	if (m_explorer->preprocess(image) == false)
	{
		Torch::message("Scanner::process - explorer failed to preprocess the image!\n");
		return false;
	}

	// Scan the image for each ROI
	for (int i = 0; i < m_n_rois; i ++)
	{
		// Initialize the explorer
		if (m_explorer->init(m_rois[i]) == false)
		{
			Torch::message("Scanner::process - failed to initialize explorer!\n");
			return false;
		}

		// ... debug
		if (verbose == true)
		{
			Torch::print("[Scanner]: ROI [%d/%d] ... \n", i + 1, m_n_rois);
		}

		// ... scanning may have multiple steps
		while (m_explorer->hasMoreSteps() == true)
		{
			// ... debug
			if (verbose == true)
			{
				Torch::print("[Scanner]: calling explorer one more time... \n");
			}

			if (m_explorer->process() == false)
			{
				Torch::message("Scanner::process - failed to run the explorer!\n");
				return false;
			}
		}

		// ... debug
		if (verbose == true)
		{
			Torch::print("[Scanner]: finished!\n");
		}

		// Select (and accumulate) the best patterns
		if (m_selector->process(m_explorer->getPatternSpace()) == false)
		{
			Torch::message("Scanner::process - failed to run the selector!\n");
			return false;
		}

		// Adjust also the scanning statistics
		m_stat_scanned += m_explorer->getNoScannedSWs();
		m_stat_prunned += m_explorer->getNoPrunnedSWs();
		m_stat_accepted += m_explorer->getNoAcceptedSWs();
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////
// Check if the scanning parameters are set OK

bool Scanner::checkParameters(const Image& image) const
{
	const int image_w = image.getWidth();
	const int image_h = image.getHeight();

	// Check the image size (and the number of color channels)
	if (image_w <= 0 || image_h <= 0)
	{
		Torch::message("Scanner::checkParameters - invalid specified image size!\n");
		return false;
	}
	if (image.getNPlanes() != 1)
	{
		Torch::message("Scanner::checkParameters - only grayscale image accepted!\n");
		return false;
	}

	// Check if any ROI and if they have the correct coordinates
	if (m_rois == 0)
	{
		Torch::message("Scanner::checkParameters - no ROI specified, maybe the image size is invalid!\n");
		return false;
	}
	for (int i = 0; i < m_n_rois; i ++)
	{
		const sRect2D& roi = m_rois[i];
		if (!	(roi.x >= 0 && roi.w > 0 && roi.y >= 0 && roi.h > 0 &&
			 roi.x + roi.w <= image_w && roi.y + roi.h <= image_h))
		{
			Torch::message("Scanner::checkParameters - invalid ROI!\n");
			return false;
		}
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////
// Access functions

const Torch::sRect2D& Scanner::getROI(int index) const
{
	// Check parameters
	if (index < 0 || index >= getNoROIs())
	{
		Torch::error("Scanner::getROI - invalid index!\n");
	}

	// OK
	return m_rois[index];
}

const PatternList& Scanner::getPatterns() const
{
	// Check parameters
	if (m_selector == 0)
	{
		Torch::error("Scanner::getPatterns - invalid selector!\n");
	}

	// OK
	return m_selector->getPatterns();
}

/////////////////////////////////////////////////////////////////////////

}
