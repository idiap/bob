#ifndef _TORCH5SPRO_LBP_MACHINE_H_
#define _TORCH5SPRO_LBP_MACHINE_H_

#include "Machine.h"	// LBPMachine is a <Machine>

namespace Torch
{
	class ipLBP;

        //////////////////////////////////////////////////////////////////////////////////////
	// Torch::LBPMachine:
	//	Process an image (or sub-window) with the size of the model.
	//	This Machine is trained with a boosting algorithm and uses <ipLBP> like features.
	//
	//	The parameters of the machine are:
	//		- the pixel location to apply the LBP on. #pixel_location#
	//		- a look-up table containing the weights for each kernel indexes. #lut#
	//
	//	Beside these parameters, in the model file there is saved (and load):
	//		- the LBP code type and radius (used for creating the the <ipLBP> object
	//					for feature extraction)
	//
	// TODO: doxygen header!
	//////////////////////////////////////////////////////////////////////////////////////

	class LBPMachine : public Machine
	{
	public:

		// Supported LBP code types
		enum LBPType
		{
			LBP4RCenter,
			LBP4RCenter_RI,
			LBP4RCenter_U2,
			LBP4RCenter_U2RI,

			LBP4RAverage,
			LBP4RAverage_RI,
			LBP4RAverage_U2,
			LBP4RAverage_U2RI,

			LBP4RAverageAddBit,

			LBP8RCenter,
			LBP8RCenter_RI,
			LBP8RCenter_U2,
			LBP8RCenter_U2RI,

			LBP8RAverage,
			LBP8RAverage_RI,
			LBP8RAverage_U2,
			LBP8RAverage_U2RI,

			LBP8RAverageAddBit,
		};

		/// Constructor
		LBPMachine(LBPType lbp_type = LBP8RAverageAddBit);

		/// Destructor
		virtual ~LBPMachine();

		// Change the LBP code type to work with
		bool			setLBPType(LBPType lbp_type);
		bool			setLBPRadius(int lbp_radius);

		// Creates an <ipLBP> associated to the given LBP code type
		static ipLBP*		makeIpLBP(LBPType lbp_type);

		// Set the machine's parameters
		bool			setLUT(double* lut, int lut_size);
		bool			setXY(int x, int y);

		// Change the input size (need to set the model size to the <ipLBP>) - overriden
		virtual void		setSize(const TensorSize& size);

		///////////////////////////////////////////////////////////

		/// Process the input tensor
		virtual bool 		forward(const Tensor& input);

		/// Loading/Saving the content from files (\emph{not the options}) - overriden
		virtual bool		loadFile(File& file);
		virtual bool		saveFile(File& file) const;

		/// Constructs an empty Machine of this kind - overriden
		/// (used by <MachineManager>, this object should be deallocated by the user)
		virtual Machine*	getAnInstance() const { return new LBPMachine(); }

		// Get the ID specific to each Machine - overriden
		virtual int		getID() const { return LBP_MACHINE_ID; }

		///////////////////////////////////////////////////////////
		// Access functions

		LBPType			getLBPType() const { return m_lbp_type; }
		int			getLBPRadius() const;
		int			getX() const { return m_x; }
		int			getY() const { return m_y; }

		///////////////////////////////////////////////////////////

	protected:

		///////////////////////////////////////////////////////////////
		// Attributes

		// Parameters
		double*			m_lut;		// The lookup table with weights for each LBP code
		int			m_lut_size;
		int			m_x, m_y;	// Position where to extract the LBP code from

		// Object to compute the required LBP features
		LBPType			m_lbp_type;
		ipLBP*			m_ip_lbp;

		// Fast access to the output value
		double*			m_fast_output;	// Pointer to the DoubleTensor
	};


        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // REGISTER this machine to the <MachineManager>
        const bool lbp_machine_registered = MachineManager::getInstance().add(
                new LBPMachine(), "LBPMachine");
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
}

#endif
