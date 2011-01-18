#ifndef _TORCH5SPRO_IP_VCYCLE_H_
#define _TORCH5SPRO_IP_VCYCLE_H_

#include "ip/ipCore.h"

#define IS_NEAR(var, value, delta) ((var >= (value - delta)) && (var <= (value + delta)))

namespace Torch {

/** This class implements the multigrid V-cycle algorithm

    The V-cycle algorithm is an iterative method allowing the resolution (approximation)
    of large and sparse linear systems induced by partial differential equations. 
    Multiple grids of different resolution are used in order to speed up the resolution. 
    Note that the number of grids is dependent on the initial size of the 2D data.
   
    \verbatim
    "A Multigrid Tutorial", W. L. Briggs. SIAM Books, Philadelphia, 1987.
    \endverbatim

    @author Guillaume Heusch (heusch@idiap.ch)
    @author Laurent El Shafey (laurent.el-shafey@idiap.ch)
    @version 2.0
    \date
    @since 2.0
*/
class ipVcycle : public ipCore
{
	public:

		// Constructor
		ipVcycle();

		// Destructor
		virtual ~ipVcycle();

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
		int n_grids_;
		int width_, height_;

		/** this function implements the v-cycle (recursive call), in order to solve 
		    the large and sparse linear system Ax=b
		      
		  @param x the result
		  @param b the right-hand side term in Ax=b
		  @param lambda relative importance of the smoothness constraint
		  @param level 'depth' of the current grid (0 = finest grid)
		  @param type type of diffusion (isotropic, anisotropic)
		*/
  		DoubleTensor* mgv(DoubleTensor& x, DoubleTensor& b, double lambda, int level, int type );

		bool cutExtremum(DoubleTensor& data, int distribution_width, int p);
};
 
}

#endif
