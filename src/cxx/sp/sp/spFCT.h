#ifndef _TORCHSPRO_SP_FCT_H_
#define _TORCHSPRO_SP_FCT_H_

#include "core/Tensor.h"
#include "sp/spCore.h"

namespace Torch
{
/**
 * \ingroup libsp_api
 * @{
 *
 */

  /**
   * @brief This class is designed to perform a DCT.
   * The output is a FloatTensor.
   */
	class spFCT : public spCore
	{
	public:
		/**
     *  Constructor
     */
		spFCT(bool inverse_ = false);

		/**
     *  Destructor
     */
		virtual ~spFCT();


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
		// Attributes
    /**
     * Indicates if we want to compute the inverse DCT
     */
		bool inverse;


		int N;
		int H, W;

		DoubleTensor *R;
	};

/**
 * @}
 */

}

#endif
