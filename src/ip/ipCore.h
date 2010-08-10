#ifndef IPCORE_INC
#define IPCORE_INC

#include "core/spCore.h"

/**
 * \addtogroup libip_api libIP API
 * @{
 *
 *  The libIP API.
 */
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

	protected:

		/////////////////////////////////////////////
		/// Attributes

		//
	};

}


/**
 * @}
 */


/**
@page libIP IP: an Image Processing module

@section intro Introduction

IP is the Image Processing module of Torch.

@section api Documentation
- @ref libip_api "libIP API"

*/

#endif
