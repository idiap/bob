#ifndef APCORE_INC
#define APCORE_INC

#include "core/spCore.h"

/**
 * \addtogroup libap_api libAP API
 * @{
 *
 *  The libAP API.
 */
namespace Torch {

	class apCore : public spCore
	{
	public:
		/// Constructor
		apCore();

		/// Destructor
		virtual ~apCore();

		/// Change the input audio size
		virtual bool		setAudioSize(int new_length);

		/// Retrieve the input audio size
		int			getAudioSize() const;

	protected:

		/////////////////////////////////////////////
		/// Attributes

		int			m_audioSize;	// Will process only inputs of this size!
	};

}

/**
 * @}
 */


/**
@page libAP AP: an Audio Processing module

@section intro Introduction

AP is the Audio Processing module of Torch.

@section api Documentation
- @ref libap_api "libAP API"

*/

#endif
