#ifndef _TORCHVISION_IP_GEOM_NORM_H_
#define _TORCHVISION_IP_GEOM_NORM_H_

#include "ipCore.h"		// <ipGeomNorm> is a <Torch::ipCore>
#include "vision.h"		// <sPoint2D> definition
#include "ipRotate.h"
#include "ipScaleYX.h"
#include "ipCrop.h"

namespace Torch
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::ipGeomNorm
	//	This class is designed to geometrically normalize a 2D/3D tensor,
	//		using some ground truth points.
	//	The normalized tensor has the same storage type and is of required size.
	//
	//	- PARAMETERS (name, type, default value, description):
	//		"rotIdx1"	integer	0	"rotation: index of the first point to define the rotation axis - RP1"
	//		"rotIdx2"	integer	0	"rotation: index of the second point to define the rotation axis - RP2"
	//		"rotAngle"	double	0.0	"rotation: desired angle of the RP1-RP2 in the normalized image"
	//		"scaleIdx1"	integer 0	"scale: index of the first point to define the scalling factor - SP1"
	//		"scaleIdx2"	integer 0	"scale: index of the second point to define the scalling factor - SP2"
	//		"scaleDist"	integer 0	"scale: desired distance (SP1, SP2) in the normalized image"
	//		"cropIdx1"	integer 0	"crop: index of the first point to define the cropping center - CP1"
	//		"cropIdx2"	integer 0	"crop: index of the second point to define the cropping center - CP2"
	//		"cropDx"	integer 0	"crop: Ox offset of the (CP1, CP2) center in the normalized image"
	//		"cropDy"	integer 0	"crop: Oy offset of the (CP1, CP2) center in the normalized image"
	//		"cropW"		integer 0	"crop: width of the base normalized image (without border)"
	//		"cropH"		integer 0	"crop: height of the base normalized image (without border)"
	//		"cropBorderX"	integer 0	"crop: Ox border of the  normalized image around the cropping center"
	//		"cropBorderY"	integer 0	"crop: Oy border of the  normalized image around the cropping center"
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class ipGeomNorm : public ipCore
	{
	public:

		// Constructor
		ipGeomNorm();

		// Destructor
		virtual ~ipGeomNorm();

		// Change the ground truth points to use for normalization
		bool			setGTPoints(const sPoint2D* gt_pts, int n_gt_pts);

		// Access the normalized points
		const sPoint2D*		getNMPoints() const { return m_nm_pts; }

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

		// Ground truth and normalized points
		const sPoint2D*		m_gt_pts;
		sPoint2D*		m_nm_pts;
		int			m_n_gt_pts;

		// <ipCore>s for rotation, scalling and cropping
		ipRotate		m_ip_rotate;
		ipScaleYX		m_ip_scale;
		ipCrop			m_ip_crop;
	};
}

#endif
