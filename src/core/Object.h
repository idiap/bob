#ifndef _TORCH5SPRO_OBJECT_H_
#define _TORCH5SPRO_OBJECT_H_

#include "core/general.h"
#include "core/VariableCollector.h"

namespace Torch
{
   	class VariableCollector;

	class Object
	{
	public:

		/// Constructor
		Object();

	    	/// Destructor
	    	virtual ~Object();

		///////////////////////////////////////////////////////////
		/// Option management functions - adding new ones

		/// returns \c false if the name is already taken
		bool			addBOption(const char* name, bool init_value, const char* help = "");
		bool			addIOption(const char* name, int init_value, const char* help = "");
		bool			addFOption(const char* name, float init_value, const char* help = "");
		bool			addDOption(const char* name, double init_value, const char* help = "");
		bool			addSOption(const char* name, const char* init_value, const char* help = "");

		///////////////////////////////////////////////////////////
		/// Option management functions - changing their values

		/// returns \c false if the name is wrong or the option has a different type
		bool			setBOption(const char* name, bool new_value);
		bool			setIOption(const char* name, int new_value);
		bool			setFOption(const char* name, float new_value);
		bool			setDOption(const char* name, double new_value);
		bool			setSOption(const char* name, const char* new_value);

		///////////////////////////////////////////////////////////
		/// Option management functions - retrieving alues

		/// it the name is wrong of the option has a different type, the return value is invalid
		///	check the \c ok value
		bool			getBOption(const char* name, bool* ok = 0);
		int			getIOption(const char* name, bool* ok = 0);
		float			getFOption(const char* name, bool* ok = 0);
		double			getDOption(const char* name, bool* ok = 0);
		const char*		getSOption(const char* name, bool* ok = 0);

	protected:

		/// called when some option was changed
		virtual void		optionChanged(const char* name) { }

	private:

		///////////////////////////////////////////////////////////
		/// Attributes

		VariableCollector*	m_optionImpl;		// Implementation details (not exposed in the interface)
	};
}

#endif
