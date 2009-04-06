#include "ipSWVariancePruner.h"
#include "Tensor.h"

namespace Torch
{
        /////////////////////////////////////////////////////////////////////////
        struct SquarePixelOperator : public ipIntegral::PixelOperator
        {
        public:
                // Compute the square of some pixel
                virtual int     compute(char px) { return px * px; }
                virtual int     compute(short px) { return px * px; }
                virtual int     compute(int px) { return px * px; }
                virtual long    compute(long px) { return px * px; }
                virtual double  compute(float px) { return px * px; }
                virtual double  compute(double px) { return px * px; }
        };
        /////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////
// Constructor

ipSWVariancePruner::ipSWVariancePruner()
	:	ipSWPruner(),
                m_square_pxOp(new SquarePixelOperator()),
                m_use_mean(true),
                m_use_stdev(true),

                m_min_mean(0.0), m_max_mean(0.0),
                m_min_stdev(0.0), m_max_stdev(0.0),

                m_sw_size(0.0),
                m_scaled_min_mean(0.0), m_scaled_max_mean(0.0),
                m_square_min_stdev(0.0), m_square_max_stdev(0.0)
{
        addBOption("UseMean", true, "prune using the mean");
        addBOption("UseStdev", true, "prune using the standard deviation");
}

/////////////////////////////////////////////////////////////////////////
// Destructor

ipSWVariancePruner::~ipSWVariancePruner()
{
        delete m_square_pxOp;
}

/////////////////////////////////////////////////////////////////////////
// called when some option was changed - overriden

void ipSWVariancePruner::optionChanged(const char* name)
{
        m_use_mean = getBOption("UseMean");
        m_use_stdev = getBOption("UseStdev");
}

/////////////////////////////////////////////////////////////////////////
/// Change the region of the input tensor to process

void ipSWVariancePruner::setRegion(const TensorRegion& region)
{
	// If the sub-window size is changed, then update the precomputed factors
        if (	m_region.size[0] != region.size[0] ||
		m_region.size[1] != region.size[1])
        {
                m_sw_size = region.size[0] * region.size[1];
                m_scaled_min_mean = m_min_mean * m_sw_size;
                m_scaled_max_mean = m_max_mean * m_sw_size;
                m_square_min_stdev = m_min_stdev * m_min_stdev * m_sw_size * m_sw_size;
                m_square_max_stdev = m_max_stdev * m_max_stdev * m_sw_size * m_sw_size;
        }

	// Set the region (subwindow) to process
	setRegion(region);
        m_isRejected = false;

        // Compute the sum and the square sum if required
        const double sum = (m_use_mean == true || m_use_stdev == true) ? getSumII(m_ipi_average) : 0.0;
        const double square_sum = (m_use_stdev == true) ? getSumII(m_ipi_square) : 0.0;

        // Prune using the mean
        if (m_use_mean == true)
        {
                m_isRejected = sum < m_scaled_min_mean || sum > m_scaled_max_mean;
        }

        // Prune using the standard deviation
        if (m_isRejected == false && m_use_stdev == true)
        {
                const double square_stdev = square_sum * m_sw_size - sum * sum;
                m_isRejected = square_stdev < m_square_min_stdev || square_stdev > m_square_max_stdev;
        }
}

/////////////////////////////////////////////////////////////////////////
// Compute the sum of values in some subwindow

double ipSWVariancePruner::getSumII(const ipIntegral& ipi)
{
	const int sw_x = m_region.pos[1];
	const int sw_y = m_region.pos[0];
	const int sw_w = m_region.size[1];
	const int sw_h = m_region.size[0];

        const Tensor& ii = ipi.getOutput(0);
        switch (ii.getDatatype())
        {
        case Tensor::Int:
                {
                        const IntTensor& data = *((IntTensor*)&ii);

                        if (data.nDimension() == 2)
                        {
                                return  (data.get(sw_y, sw_x) + data.get(sw_y + sw_h, sw_x + sw_w)) -
                                        (data.get(sw_y + sw_h, sw_x) + data.get(sw_y, sw_x + sw_w));
                        }
                        else
                        {
                                return  (data.get(sw_y, sw_x, 0) + data.get(sw_y + sw_h, sw_x + sw_w, 0)) -
                                        (data.get(sw_y + sw_h, sw_x, 0) + data.get(sw_y, sw_x + sw_w, 0));
                        }
                }
                break;

        case Tensor::Long:
                {
                        const LongTensor& data = *((LongTensor*)&ii);

                        if (data.nDimension() == 2)
                        {
                                return  (data.get(sw_y, sw_x) + data.get(sw_y + sw_h, sw_x + sw_w)) -
                                        (data.get(sw_y + sw_h, sw_x) + data.get(sw_y, sw_x + sw_w));
                        }
                        else
                        {
                                return  (data.get(sw_y, sw_x, 0) + data.get(sw_y + sw_h, sw_x + sw_w, 0)) -
                                        (data.get(sw_y + sw_h, sw_x, 0) + data.get(sw_y, sw_x + sw_w, 0));
                        }
                }
                break;

        case Tensor::Double:
                {
                        const DoubleTensor& data = *((DoubleTensor*)&ii);

                        if (data.nDimension() == 2)
                        {
                                return  (data.get(sw_y, sw_x) + data.get(sw_y + sw_h, sw_x + sw_w)) -
                                        (data.get(sw_y + sw_h, sw_x) + data.get(sw_y, sw_x + sw_w));
                        }
                        else
                        {
                                return  (data.get(sw_y, sw_x, 0) + data.get(sw_y + sw_h, sw_x + sw_w, 0)) -
                                        (data.get(sw_y + sw_h, sw_x, 0) + data.get(sw_y, sw_x + sw_w, 0));
                        }
                }
                break;

        default:
                Torch::error("ipSWVariancePruner::setSubWindow - invalid integral type!\n");
                return 0.0;
        }
}

/////////////////////////////////////////////////////////////////////////
// Check if the input tensor has the right dimensions and type

bool ipSWVariancePruner::checkInput(const Tensor& input) const
{
	// Check parameters
        return input.nDimension() == 2 || input.nDimension() == 3;
}

/////////////////////////////////////////////////////////////////////////
// Allocate (if needed) the output tensors given the input tensor dimensions

bool ipSWVariancePruner::allocateOutput(const Tensor& input)
{
        // No output needed!
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Process some input tensor (the input is checked, the outputs are allocated)

bool ipSWVariancePruner::processInput(const Tensor& input)
{
        // Compute the integral image
        if (m_ipi_average.process(input) == false)
        {
                Torch::message("ipSWVariancePruner::processInput - failed to compute the integral!\n");
                return false;
        }

        // Compute the square integral image
        m_ipi_square.setPixelOperator(m_square_pxOp);
        if (m_ipi_square.process(input) == false)
        {
                Torch::message("ipSWVariancePruner::processInput - failed to compute the square integral!\n");
                return false;
        }

        // OK
        return true;
}

/////////////////////////////////////////////////////////////////////////

}
