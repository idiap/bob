#ifndef _TORCHVISION_IP_GEOM_NORM_H_
#define _TORCHVISION_IP_GEOM_NORM_H_

#include "ipCore.h"		// <ipGeomNorm> is a <Torch::ipCore>
#include "vision.h"		// <sPoint2D> definition
#include "ipRotate.h"
#include "ipScaleYX.h"
#include "ipCrop.h"

namespace Torch
{
	class GTFile;

	/////////////////////////////////////////////////////////////////////////
	// Torch::ipGeomNorm
	//	This class is designed to geometrically normalize a 2D/3D tensor,
	//		using some ground truth points.
	//	The normalized tensor has the same storage type and is of required size.
	//
	//	- PARAMETERS (name, type, default value, description):
	//		"rotPoint1"	string	""	"rotation: first point to define the rotation axis - RP1"
	//		"rotPoint2"	string	""	"rotation: second point to define the rotation axis - RP2"
	//		"rotAngle"	double	0.0	"rotation: desired angle of the RP1-RP2 in the normalized image"
	//		"scalePoint1"	string 	""	"scale: first point to define the scalling factor - SP1"
	//		"scalePoint2"	string 	""	"scale: second point to define the scalling factor - SP2"
	//		"scaleDist"	integer 0	"scale: desired distance (SP1, SP2) in the normalized image"
	//		"cropPoint1"	string 	""	"crop: first point to define the cropping center - CP1"
	//		"cropPoint2"	string 	""	"crop: second point to define the cropping center - CP2"
	//		"cropDx"	integer 0	"crop: Ox offset of the (CP1, CP2) center in the normalized image"
	//		"cropDy"	integer 0	"crop: Oy offset of the (CP1, CP2) center in the normalized image"
	//		"cropW"		integer 0	"crop: width of the base normalized image (without border)"
	//		"cropH"		integer 0	"crop: height of the base normalized image (without border)"
	//		"cropBorderX"	integer 0	"crop: Ox border of the  normalized image around the cropping center"
	//		"cropBorderY"	integer 0	"crop: Oy border of the  normalized image around the cropping center"
	//		"finalRotAngle"	double	0.0	"final rotation: final rotation angle of the center"
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

		// Load the configuration parameters from a text file
		bool			loadCfg(const char* filename);

		// Change the ground truth points to use for normalization
		bool			setGTFile(const GTFile* gt_file);

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
		const GTFile*		m_gt_file;
		sPoint2D*		m_nm_pts;

		// <ipCore>s for rotation, scalling and cropping
		ipRotate		m_ip_rotate;
		ipScaleYX		m_ip_scale;
		ipCrop			m_ip_crop;
	};
}

#endif
