#ifndef _TORCHSPRO_SP_DFT_H_
#define _TORCHSPRO_SP_DFT_H_

#include "core/Tensor.h"
#include "sp/spCore.h"

namespace Torch {

/**
 * \ingroup libsp_api
 * @{
 *
 */

  /**
   * @brief A naive implementation of the Discrete Fourier Transform
   * The output is a FloatTensor
   */
	class spDFT : public spCore
	{
	public:
		/** 
     * Constructor
     */
		spDFT(bool inverse_ = false);

		/** 
     * Destructor
     */
		virtual ~spDFT();


	protected:
		/**
     *  Check if the input tensor has the right dimensions - overriden
     */
		virtual bool		checkInput(const Tensor& input) const;

		/**
     *  Allocate (if needed) the output tensors given the input tensor 
     *  dimensions - overriden
     */
		virtual bool		allocateOutput(const Tensor& input);

		/**
     *  Process some input tensor (the input is checked, the outputs are 
     *  allocated) - overriden
     */
		virtual bool		processInput(const Tensor& input);


	private:
    /**
     * Initialize an array of exponential coefficients used to compute 
     * the DFT
     */
    bool initExpArray(const int NN);

    /**
     * Initialize two arrays of exponential coefficients used to compute
     * the 2D DFT
     */
    bool initExpArray(const int HH, const int WW);


		// Attributes
    /**
     * Indicates if the direct or inverse DFT should be performed
     */
		bool inverse;

    /**
     * The full size of the signal
     */
		int N;

    /**
     * The dimensions of a 2D signal (for the 2D case only)
     */
		int H,W;

    /**
     * DoubleTensor for the input signal
     */
		DoubleTensor *sig;

    /**
     * First array of exponential coefficients
     */
    DoubleTensor *exp_coef1;

    /**
     * Second array of exponential coefficients (for the 2D case only)
     */
    DoubleTensor *exp_coef2;
	};

/**
 * @}
 */

}

#endif

