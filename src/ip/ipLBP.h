#ifndef _TORCHVISION_IP_LBP_H_
#define _TORCHVISION_IP_LBP_H_

#include "ipCore.h"		// <ipLBP> is an <ipCore>

namespace Torch
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::ipLBP
	//	This class computes the LBP code at a given location in the image.
	//	The result is a 1D IntTensor with a single value (the LBP code).
	//
	//	- PARAMETERS (name, type, default value, description):
	//		"ToAverage"	bool	false	"compute the LBP code to the average"
	//		"AddAvgBit"	bool	false	"add to the LBP code the center (considering the average)"
	//		"Uniform"	bool	false	"uniform patterns (at most two bitwise 0-1 or 1-0 transitions)"
	//		"RotInvariant"	bool	false	"rotation invariant patterns"
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class ipLBP : public ipCore
	{
	public:

		// Constructor
		ipLBP(int P, int R = 1);

		// Destructor
		virtual ~ipLBP();

		// Set the LBP location
		bool			setXY(int x, int y);

		// Set the radius value of the LBP operator
		bool			setR(int R);

		// Get the maximum possible label
		virtual int		getMaxLabel() = 0;

		// Get the radius value of the LBP operator
		int			getR() { return m_R; };

		// Get the LBP code (fast & direct) access
		int			getLBP() const { return *m_lbp; }

		/////////////////////////////////////////////////////////////////

	protected:

		//////////////////////////////////////////////////////////

		/// Check if the input tensor has the right dimensions and type - overriden
		virtual bool		checkInput(const Tensor& input) const;

		/// Allocate (if needed) the output tensors given the input tensor dimensions - overriden
		virtual bool		allocateOutput(const Tensor& input);

		/// called when some option was changed - overriden
		virtual void		optionChanged(const char* name);

		// Bilinear interpolation
		short			bilinear_interpolation(	const short* src,
								int stride_w, int stride_h,
								float x, float y);

		//////////////////////////////////////////////////////////

	protected:

		/////////////////////////////////////////////////////////////////
		// Attributes

		// LBP operator parameters
		int			m_P, m_R;

		// LBP operator location
		int			m_x, m_y;

		// Direct (&fast) access to the LBP code
		int*			m_lbp;

		// Different parameters for computing LBPs (not using <getOption> to speed up computation)
		bool			m_toAverage;
		bool			m_addAvgBit;
		bool			m_uniform;
		bool			m_rot_invariant;
	};
}

#endif



