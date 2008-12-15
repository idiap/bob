#ifndef _TORCHVISION_SCANNING_IP_SW_VARIANCE_PRUNER_H_
#define _TORCHVISION_SCANNING_IP_SW_VARIANCE_PRUNER_H_

#include "ipSWPruner.h"		// <ipSWVariancePruner> is a <ipSWPruner>
#include "ipIntegral.h"

namespace Torch
{
        class Tensor;
        struct SquarePixelOperator;

	/////////////////////////////////////////////////////////////////////////
	// Torch::ipSWVariancePruner
	//	- rejects a sub-window based on the pixel/edge variance
	//              (generally too smooth sub-windows)
	//
	//	- PARAMETERS (name, type, default value, description):
	//		"UseMean"	bool    true    "prune using the mean"
	//              "UseStdev"      bool    true    "prune using the standard deviation"
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class ipSWVariancePruner : public ipSWPruner
	{
	public:

		// Constructor
		ipSWVariancePruner();

		// Destructor
		virtual ~ipSWVariancePruner();

		/// Change the sub-window to process in - overriden
		/// Checks also if the sub-window is rejected
		virtual bool		setSubWindow(int sw_x, int sw_y, int sw_w, int sw_h);

		// Set prunning parameters
		double                  setMinMean(double min_mean) { m_min_mean = min_mean; }
		double                  setMaxMean(double max_mean) { m_max_mean = max_mean; }
		double                  setMinStdev(double min_stdev) { m_min_stdev = min_stdev; }
		double                  setMaxStdev(double max_stdev) { m_max_stdev = max_stdev; }

        protected:

                /////////////////////////////////////////////////////////////////

                /// Check if the input tensor has the right dimensions and type - override
		virtual bool		checkInput(const Tensor& input) const;

		/// Allocate (if needed) the output tensors given the input tensor dimensions - override
		virtual bool		allocateOutput(const Tensor& input);

		/// Process some input tensor (the input is checked, the outputs are allocated) - override
		virtual bool		processInput(const Tensor& input);

		/// called when some option was changed - overriden
		virtual void		optionChanged(const char* name);

		/////////////////////////////////////////////////////////////////

        private:

                // Compute the sum of values in some subwindow
                double                  getSumII(const ipIntegral& ipi);

		/////////////////////////////////////////////////////////////////
		// Attributes

		// <ipIntegral>s to compute fast the stdev at any sub-window
		ipIntegral              m_ipi_average;
		ipIntegral              m_ipi_square;
                SquarePixelOperator*    m_square_pxOp;   // Operator used by the square integral image

                // Parameters for checking the mean and standard deviation in some subwindow
                bool                    m_use_mean;     // Mean/Stdev switches
                bool                    m_use_stdev;
                double                  m_min_mean;     // Min/max mean
                double                  m_max_mean;
                double                  m_min_stdev;    // Min/max stdev
                double                  m_max_stdev;

                // Precomputed values
                double                  m_sw_size;              // (sw_w * sw_h)
                double                  m_scaled_min_mean;      // mean * (sw_w * sw_h)
                double                  m_scaled_max_mean;
                double                  m_square_min_stdev;     // stdev * stdev * (sw_w * sw_h) * (sw_w * sw_h)
                double                  m_square_max_stdev;
	};
}

#endif
