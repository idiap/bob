#ifndef _TORCH5SPRO_OBJECT_H_
#define _TORCH5SPRO_OBJECT_H_

#include "general.h"

namespace Torch
{
	class OptionImpl;

	/** Almost all classes in Torch should be a sub-class of this class.
	    It provides a useful interface to manage options and to load and save
	    its content from/to files.

	    By default, the following options are added to any <Object>:
		-	"verbose", boolean, default value = [false], "verbose flag"

	    @author Ronan Collobert (collober@idiap.ch)
	*/
	class Object
	{
	public:

		/// Constructor
		Object();

	    	/// Destructor
	    	virtual ~Object();

		///////////////////////////////////////////////////////////
		/// Option management functions - adding new ones

		/// returns <false> if the name is already taken
		bool			addBOption(const char* name, bool init_value, const char* help = "");
		bool			addIOption(const char* name, int init_value, const char* help = "");
		bool			addFOption(const char* name, float init_value, const char* help = "");
		bool			addDOption(const char* name, double init_value, const char* help = "");

		///////////////////////////////////////////////////////////
		/// Option management functions - changing their values

		/// returns <false> if the name is wrong or the option has a different type
		bool			setBOption(const char* name, bool new_value);
		bool			setIOption(const char* name, int new_value);
		bool			setFOption(const char* name, float new_value);
		bool			setDOption(const char* name, double new_value);

		///////////////////////////////////////////////////////////
		/// Option management functions - retrieving alues

		/// it the name is wrong of the option has a different type, the return value is invalid
		///	check the <ok> value
		bool			getBOption(const char* name, bool* ok = 0);
		int			getIOption(const char* name, bool* ok = 0);
		float			getFOption(const char* name, bool* ok = 0);
		double			getDOption(const char* name, bool* ok = 0);

	protected:

		/// called when some option was changed
		virtual void		optionChanged(const char* name) { }

	private:

		///////////////////////////////////////////////////////////
		/// Attributes

		OptionImpl*		m_optionImpl;		// Implementation details (not exposed in the interface)
	};
}

#endif
