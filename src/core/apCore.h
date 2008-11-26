#ifndef APCORE_INC
#define APCORE_INC

#include "spCore.h"

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

#endif
