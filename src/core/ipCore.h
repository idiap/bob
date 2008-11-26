#ifndef IPCORE_INC
#define IPCORE_INC

#include "spCore.h"
#include "vision.h"		// <sSize> definition

namespace Torch
{
	//////////////////////////////////////////////////////////////////////////////////////
	// Torch::ipCore:
	//	- image processing interface
	//
	// TODO: doxygen header!
	//////////////////////////////////////////////////////////////////////////////////////

	class ipCore : public spCore
	{
	public:
		/// Constructor
		ipCore();

		/// Destructor
		virtual ~ipCore();

		/// Change the input image size
		bool			setInputSize(const sSize& new_size);
		bool			setInputSize(int new_w, int new_h);

		/// Retrieve the input image size
		int			getInputWidth() const;
		int			getInputHeight() const;

	protected:

		/////////////////////////////////////////////
		/// Attributes

		sSize			m_inputSize;	// Will process only inputs of this size!s
	};

}

#endif
