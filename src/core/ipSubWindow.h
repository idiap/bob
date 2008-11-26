#ifndef IPCORE_SUB_WINDOW_INC
#define IPCORE_SUB_WINDOW_INC

#include "ipCore.h"

namespace Torch {

	/// ipSubWindow - process some sub-window of the image
	class ipSubWindow : public ipCore
	{
	public:
		/// Constructor
		ipSubWindow();

		/// Destructor
		virtual ~ipSubWindow();

		/// Change the sub-window to process in
		virtual bool		setSubWindow(int sw_x, int sw_y, int sw_w, int sw_h);

		/// Change the input image size - overriden (it will reset the sub-window)
		virtual bool		setInputSize(const sSize& new_size);
		virtual bool		setInputSize(int new_w, int new_h);

		/// Retrieve the sub-window to process in
		int			getSubWindowX() const;
		int			getSubWindowY() const;
		int			getSubWindowW() const;
		int			getSubWindowH() const;

	protected:

		/////////////////////////////////////////////
		/// Attributes

		int			m_sw_x, m_sw_y,	// Sub-window's positions
					m_sw_w, m_sw_h;	// ... and its size
	};

}

#endif
