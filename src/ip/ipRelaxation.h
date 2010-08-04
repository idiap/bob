#ifndef _TORCH5SPRO_IP_RELAXATION_H_
#define _TORCH5SPRO_IP_RELAXATION_H_

#include "core/ipCore.h"

#define IS_NEAR(var, value, delta) ((var >= (value - delta)) && (var <= (value + delta)))

namespace Torch {

/** This class is designed to apply relaxation on the linear system induced by
    the discretization of an elliptic PDE (diffusion)

    Relaxation is an iterative method allowing the resolution (approximation)
    of large and sparse linear systems. Here, the Gauss-Seidel scheme with 
    red-black ordering is used (see multigrid.h file). The number of relaxation 
    steps can be provided (default 10).

    @author Guillaume Heusch (heusch@idiap.ch)
    @version 2.0
    \Date
    @since 2.0
*/

class ipRelaxation : public ipCore
{
	public:

		// Constructor
		ipRelaxation();

		// Destructor
		virtual ~ipRelaxation();

	protected:

		//////////////////////////////////////////////////////////

		/// Check if the input tensor has the right dimensions and type - overriden
		virtual bool		checkInput(const Tensor& input) const;

		/// Allocate (if needed) the output tensors given the input tensor dimensions - overriden
		virtual bool		allocateOutput(const Tensor& input);

		/// Process some input tensor (the input is checked, the outputs are allocated) - overriden
		virtual bool		processInput(const Tensor& input);

		//////////////////////////////////////////////////////////

	private:
		bool cutExtremum(DoubleTensor& data, int distribution_width);


};

}
#endif  

  

