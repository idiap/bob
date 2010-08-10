#ifndef _TORCH5SPRO_IP_LBPBITMAP_H_
#define _TORCH5SPRO_IP_LBPBITMAP_H_

#include "ip/ipCore.h"		// <ipLBPBitmap> is an <ipCore>
#include "sp/spCores.h"

namespace Torch
{

	//================================================================================================
	//
	//Torch::ipLBPBitmap
	//	This class is designed as a preprocessing for the calculation of HLBP features.
	//      It should be associated with one particular ipLBP object. Let N_L be the number of possible labels of this ipLBP.
	//
	//	The input is an image (2D ShortTensor for now, but may be generalized to 3D ShortTensor later).
	//
	//	In the first step, the LBP image of this image is calculated.
	//
	//	In the 2nd step, a 3D ShortTensor is created having the same width and height as the LBP image, but with N_L number of frames in
	//	the 3rd dimension.
	//
	//	To fill this 3D ShortTensor, the following logic is used :
	//		A pixel(x,y) in the k-th frame is set to 1 if the LBP label of the pixel(x,y) in the LBP image is k, or else it is set to 0.
	//
	//	This 3D ShortTensor is the resulting output of this ip.
	//
	//	Comment : If this output is given to ipIntegral, and then to ipHaarLeinhart, the HLBP features will be calculated instead of the Haar
	//	features. So, this is a preprocessing to HLBP calculation.
	//	TODO: doxygen header!
	//
	//==================================================================================================

	class ipLBP;

	class ipLBPBitmap : public ipCore
	{
	public:

		// Constructor
		ipLBPBitmap(ipLBP* ip_lbp = 0);

		// Destructor
		virtual ~ipLBPBitmap();

		// Get the ID specific to each spCore - overriden
		virtual int		getID() const { return IP_LBP_BITMAP_ID; }

		/// Constructs an empty spCore of this kind - overriden
		/// (used by \c spCoreManager, this object is automatically deallocated)
		virtual spCore*		getAnInstance() const { return manage(new ipLBPBitmap()); }

		// Change the ipLBP to use
		bool			setIpLBP(ipLBP* ip_lbp);

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

		/////////////////////////////////////////////////////////////////
		// Attributes

		// The ipLBP
		ipLBP*			m_ip_lbp;
	};
}

#endif
